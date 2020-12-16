#include <cmath>
#include <map>
#include <mpi.h>
#include <string>
#include <sys/time.h>
#include <vector>

#include "data/sparse_dataset.h"
#include "logging/simple_logging.h"
#include "math/simple_algebra.h"
#include "optimizer/lr_tron_optimizer.h"
#include "other/properties.h"

std::map<int, double> compressedSparseVector(const double *vector,
                                             const int dimension) {
  std::map<int, double> index_value_map;
  for (int i = 0; i < dimension; i++) {
    if (vector[i] != 0) {
      index_value_map[i] = vector[i];
    }
  }
  return index_value_map;
}

double Predict(const double *z, SparseDataset *test_data) {
  int counter = 0;
  int sample_num = test_data->GetSampleNumber();
  for (int i = 0; i < sample_num; ++i) {
    double temp = 1.0 / (1 + exp(-1 * Dot(z, test_data->GetSample(i))));
    if (test_data->GetLabel(i) == 1 && temp >= 0.5) {
      ++counter;
    }
    if (test_data->GetLabel(i) == -1 && temp < 0.5) {
      ++counter;
    }
  }
  return counter * 100.0 / sample_num;
}

double ObjectiveValue(const double *z, SparseDataset *test_data) {
  double sum = 0;
  int sample_num = test_data->GetSampleNumber();
  for (int i = 0; i < sample_num; ++i) {
    sum += std::log(1 + std::exp(-test_data->GetLabel(i) *
                                 Dot(z, test_data->GetSample(i))));
  }
  return sum;
}

bool CheckWorkerDelay(std::map<int, int> &worker_delay, int max_delay) {
  for (auto &it : worker_delay) {
    if (it.second >= max_delay) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  int id, worker_number;
  MPI_Init(nullptr, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &worker_number);
  Properties properties(argc, argv);
  int dimension = properties.GetInt("dim");
  double rho = properties.GetDouble("rho");
  /* 需要有一个节点作为master，因此worker_num减一 */
  worker_number -= 1;
  LOG(DEBUG) << id << " go" << std::endl;
  if (id == worker_number) {
    printf("%3s %10s %10s %10s %10s %10s %10s %10s\n", "#", "r norm", "esp_pri",
           "s norm", "esp_dual", "obj_val", "accuracy", "time");
    int k = 0;
    int min_barrier = properties.GetInt("min_barrier");
    int max_delay = properties.GetInt("max_delay");
    int max_iterations = properties.GetInt("max_iter_num");
    std::string test_data_path = properties.GetString("test_data_path");
    SparseDataset test_data(test_data_path);
    double l2reg = properties.GetDouble("l2reg");
    double ABSTOL = properties.GetDouble("ABSTOL");
    double RELTOL = properties.GetDouble("RELTOL");
    double objective_value = properties.GetDouble("objective_value");
    double accuracy = properties.GetDouble("accuracy");
    auto *z = new double[dimension];
    auto *z_old = new double[dimension];
    auto *rho_x_plus_y = new double[dimension];
    std::vector<int> ready_worker_list;
    std::map<int, int> worker_delay;
    // worker_id - [x,y]
    std::map<int, std::map<int, double>> worker_x_map;
    std::map<int, std::map<int, double>> worker_y_map;
    for (int i = 0; i < worker_number; ++i) {
      //      ptr[i] = new double[dimension * 2];
      //      FillZero(worker_x_map[i], dimension * 2);
      worker_delay[i] = 0;
    }
    FillZero(z, dimension);
    LOG(INFO) << "master完成初始化";
    MPI_Barrier(MPI_COMM_WORLD);

    // status主要显示接收函数的各种错误状态
    MPI_Status status;
    //初始化两个结构体
    timeval start_time{}, end_time{};
    gettimeofday(&start_time, nullptr);
    while (true) {
      // MPI_Probe()函数探测接收消息的内容，但不影响实际接收到的消息
      MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
      int worker_id = status.MPI_SOURCE;
      ready_worker_list.push_back(worker_id);
      worker_delay[worker_id] = -1;
      auto *x_and_y = new double[dimension * 2];
      MPI_Recv(x_and_y, dimension * 2, MPI_DOUBLE, worker_id, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      worker_x_map[worker_id] = compressedSparseVector(x_and_y, dimension);
      worker_y_map[worker_id] =
          compressedSparseVector(x_and_y + dimension, dimension);
      delete[] x_and_y;
      LOG(DEBUG) << "Receive message from " << worker_id << std::endl;
      if (ready_worker_list.size() >= min_barrier &&
          CheckWorkerDelay(worker_delay, max_delay)) {
        FillZero(rho_x_plus_y, dimension);
        LOG(INFO) << "达到更新条件";
        Assign(z_old, z, dimension);
        assert(worker_x_map.size() == worker_y_map.size());
        for (auto &it : worker_x_map) {
          const int current_worker_id = it.first;
          std::map<int, double> &x_temp = it.second;
          std::map<int, double> &y_temp = worker_y_map[current_worker_id];
          for (int j = 0; j < dimension; ++j) {
            rho_x_plus_y[j] += (rho * x_temp[j] + y_temp[j]);
          }
        }
        // l2范数
        double temp = 2 * l2reg + worker_number * rho;
        for (int i = 0; i < dimension; ++i) {
          //对z更新需要求解二次型问题，得到一个解析解
          z[i] = rho_x_plus_y[i] / temp;
        }
        ++k;
        /* 完成z更新后，判断算法是否停止 */
        double nxstack = 0;
        double nystack = 0;
        double prires = 0;
        /* 计算并累加||x_i||_2^2), ||y_i||_2^2), ||r_i||_2^2) */
        assert(worker_x_map.size() == worker_y_map.size());
        for (auto &it : worker_x_map) {
          const int current_worker_id = it.first;
          std::map<int, double> &x_temp = it.second;
          std::map<int, double> &y_temp = worker_y_map[current_worker_id];
          for (int j = 0; j < dimension; ++j) {
            nxstack += x_temp[j] * x_temp[j];
            nystack += y_temp[j] * y_temp[j];
            //\|x_i-z^k\|_2^2;
            prires += (x_temp[j] - z[j]) * (x_temp[j] - z[j]);
          }
        }
        /* sqrt(sum ||x_i||_2^2) */ //\|x_i\|_2
        nxstack = sqrt(nxstack);
        /* sqrt(sum ||y_i||_2^2) */ //\|y_i\|_2
        nystack = sqrt(nystack);
        /* sqrt(sum ||r_i||_2^2) */ //\x_i-z^k\|_2
        prires = sqrt(prires);
        /* 存放||z_new - z_old||_2^2 */
        double z_diff = 0;
        /* 存放||z_new||_2^2 */
        double z_norm = 0;
        for (int i = 0; i < dimension; ++i) {
          z_diff += (z_old[i] - z[i]) * (z_old[i] - z[i]);
          z_norm += z[i] * z[i];
        }
        double dualres = rho * sqrt(worker_number * z_diff); //对偶可行性
        double eps_pri =
            sqrt(dimension * worker_number) * ABSTOL +
            RELTOL * fmax(nxstack, sqrt(worker_number * z_norm)); //原始残差
        double eps_dual = sqrt(dimension * worker_number) * ABSTOL +
                          RELTOL * nystack; //对偶残差
        gettimeofday(&end_time, nullptr);
        double wait_time = (end_time.tv_sec - start_time.tv_sec) +
                           (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        double temp_ov = ObjectiveValue(z, &test_data);
        double temp_ac = Predict(z, &test_data);
        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", k,
               prires, eps_pri, dualres, eps_dual, temp_ov, temp_ac, wait_time);
        bool can_stop = (prires <= eps_pri && dualres <= eps_dual);
        // can_stop = std::abs(temp_ov - objective_value) < 1e-2 || temp_ov <
        // objective_value
        //           || std::abs(temp_ac - accuracy) < 1e-3 || temp_ac >
        //           accuracy;
        // if (min_barrier == worker_num) {
        //    can_stop = prires <= eps_pri && dualres <= eps_dual;
        //}
        if (can_stop || k >= max_iterations) {
          for (int &it : ready_worker_list) {
            MPI_Send(nullptr, 0, MPI_INT, it, 3, MPI_COMM_WORLD);
          }
          int count = ready_worker_list.size();
          while (count < worker_number) {
            MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            worker_id = status.MPI_SOURCE;
            auto *x_and_y = new double[dimension * 2];
            MPI_Recv(x_and_y, dimension * 2, MPI_DOUBLE, worker_id, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            worker_x_map[worker_id] =
                compressedSparseVector(x_and_y, dimension);
            worker_y_map[worker_id] =
                compressedSparseVector(x_and_y + dimension, dimension);
            delete[] x_and_y;
            MPI_Send(nullptr, 0, MPI_INT, worker_id, 3, MPI_COMM_WORLD);
            ++count;
          }
          break;
        } else {
          /* 对于那些参加本次迭代并在等待中的worker，发送最新的z */
          for (int &it : ready_worker_list) {
            LOG(DEBUG) << "Send latest Z to worker: " << it << std::endl;
            MPI_Send(z, dimension, MPI_DOUBLE, it, 2, MPI_COMM_WORLD);
          }
          ready_worker_list.clear();
          for (auto &it : worker_delay) {
            ++it.second;
          }
        }
      }
    }
    double recv_buf[] = {0, 0};
    MPI_Reduce(MPI_IN_PLACE, recv_buf, 2, MPI_DOUBLE, MPI_SUM, id,
               MPI_COMM_WORLD);
    std::cout << "平均计算时间：" << recv_buf[0] / worker_number
              << "，平均等待时间：" << recv_buf[1] / worker_number << std::endl;
    delete[] z_old;
    delete[] z;
    delete[] rho_x_plus_y;
  } else {
    int master = worker_number;
    double cal_time = 0, wait_time = 0;
    auto *x = new double[dimension * 3];
    auto *y = x + dimension;
    auto *z = x + 2 * dimension;
    FillZero(x, dimension);
    FillZero(y, dimension);
    FillZero(z, dimension);

    std::string train_data_path = properties.GetString("train_data_path");
    char real_path[50];

    sprintf(real_path, train_data_path.c_str(), id);
    LOG(DEBUG) << "Worker " << id << " get train data path from: " << real_path
               << std::endl;
    SparseDataset train_data(real_path);
    LRTronOptimizer optimizer(y, z, dimension, rho, 1000, 1e-4, 0.1,
                              &train_data);
    MPI_Barrier(MPI_COMM_WORLD);

    timeval start_time{}, end_time{};
    timeval cal_start_time{}, cal_end_time{};
    gettimeofday(&start_time, nullptr);
    MPI_Status status;
    while (true) {
      gettimeofday(&cal_start_time, nullptr);
      optimizer.Optimize(x);
      gettimeofday(&cal_end_time, nullptr);
      cal_time += ((cal_end_time.tv_sec - cal_start_time.tv_sec) +
                   (cal_end_time.tv_usec - cal_start_time.tv_usec) / 1000000.0);
      MPI_Send(x, 2 * dimension, MPI_DOUBLE, master, 1, MPI_COMM_WORLD);
      LOG(DEBUG) << "Worker " << id << " send x to master" << std::endl;
      MPI_Probe(master, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == 2) {
        MPI_Recv(z, dimension, MPI_DOUBLE, master, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        LOG(DEBUG) << "Worker " << id << " receive z from master" << std::endl;
      } else if (status.MPI_TAG == 3) {
        MPI_Recv(nullptr, 0, MPI_INT, master, 3, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        break;
      }
      for (int i = 0; i < dimension; ++i) {
        y[i] += rho * (x[i] - z[i]);
      }
    }
    gettimeofday(&end_time, nullptr);
    wait_time = (end_time.tv_sec - start_time.tv_sec) +
                (end_time.tv_usec - start_time.tv_usec) / 1000000.0 - cal_time;
    double send_buf[] = {cal_time, wait_time};
    MPI_Reduce(send_buf, nullptr, 2, MPI_DOUBLE, MPI_SUM, master,
               MPI_COMM_WORLD);
    delete[] x;
  }
  MPI_Finalize();
  return 0;
}
