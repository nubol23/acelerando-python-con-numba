#include <iostream>
#include <vector>
#include <rapidcsv.h>
#include <random>
#include <indicators.hpp>

using namespace std;
using namespace indicators;

typedef vector<vector<int>> vvi;
typedef vector<int> vi;
typedef vector<double> vd;
typedef vector<vector<double>> vvd;

ProgressBar create_bar(const int n_iters) {
    return indicators::ProgressBar(
            option::BarWidth{50},
            option::Start{" ["},
            option::Fill{"="},
            option::Lead{">"},
            option::Remainder{"."},
            option::End{"]"},
            option::ForegroundColor{Color::white},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::MaxProgress{n_iters},
            option::ShowPercentage{true},
            option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    );
}

void fill_vector(vd &vec, 
                 default_random_engine &generator_,
                 normal_distribution<> &distribution)
{
    for(double &e : vec){
        e = distribution(generator_);
    }
}

void fill_mat(vvd &mat,
              default_random_engine &generator_,
              normal_distribution<> &distribution)
{
    for (vd &vec: mat) {
        fill_vector(vec, generator_, distribution);
    }
}

double dot_product(vd &vec_1, vd &vec_2) {
    size_t size = vec_1.size();

    double res = 0;
    for (int idx = 0; idx < size; idx++) {
        res += vec_1[idx] * vec_2[idx];
    }
    return res;
}

double clip(double val, double min, double max) {
    if (val < min)
        return min;
    if (max < val)
        return max;
    return val;
}

vd predict_batch(vvi &ui_mat, double mean, vd &bu, vd &bi, vvd &P, vvd &Q) {
    size_t n_samples = ui_mat[0].size();
    vd predictions = vd(n_samples);

    int u, i;
    for (int idx = 0; idx < n_samples; idx++) {
        u = ui_mat[0][idx];
        i = ui_mat[1][idx];

        predictions[idx] = clip(mean + bu[u] + bi[i] + dot_product(P[u], Q[i]), 1, 5);
    }

    return predictions;
}

double rmse(vi &vec_1, vd &vec_2) {
    double sum = 0;
    auto size = (double)vec_1.size();

    double temp;
    for (int idx = 0; idx < size; idx++) {
        temp = vec_1[idx] - vec_2[idx];
        sum += temp*temp;
    }

    return sqrt(1/size * sum);
}

tuple<double, vd, vd, vvd, vvd> fit_svd(vvi &train_data,
             vvi &val_data, 
             int n_users, 
             int n_items, 
             int k,
             double alpha_1,
             double alpha_2,
             double alpha_3,
             double alpha_4,
             double lambda_1,
             double lambda_2,
             int n_iters)
{
    random_device device_random_;
    default_random_engine generator_(device_random_());
    normal_distribution<> distribution(0, 0.1);

    vd bu(n_users, 0);
    vd bi(n_items, 0);
    vvd P(n_users, vd(k));
    vvd Q(n_items, vd(k));

    fill_mat(P, generator_, distribution);
    fill_mat(Q, generator_, distribution);

    double mean = 0;
    for (int r: train_data[2])
        mean += r;
    mean /= (double)train_data[2].size();

    double loss, min_val_loss = numeric_limits<double>::max();
    size_t n_samples = train_data[0].size();

    auto bar = create_bar(n_iters);
    for (int it = 0; it < n_iters; it++) {
        // Train step
        loss = 0;
        int u, i, r;
        double pred, error;
        for (int idx = 0; idx < n_samples; idx++) {
            u = train_data[0][idx];
            i = train_data[1][idx];
            r = train_data[2][idx];

            pred = mean + bu[u] + bi[i] + dot_product(P[u], Q[i]);

            error = (double) r - pred;

            // updating
            bu[u] += alpha_1 * (error - lambda_1 * bu[u]);
            bi[i] += alpha_2 * (error - lambda_1 * bi[i]);

            double pu, qi;
            for (int f = 0; f < k; f++) {
                pu = P[u][f];
                qi = Q[i][f];
                P[u][f] += alpha_3 * (error * qi - lambda_2 * pu);
                Q[i][f] += alpha_4 * (error * pu - lambda_2 * qi);
            }

            loss += error * error;
        }
        loss = sqrt(loss/(double) train_data[0].size());

        // Val step
        vd val_preds = predict_batch(val_data, mean, bu, bi, P, Q);
        min_val_loss = min(min_val_loss, rmse(val_data[2], val_preds));

        bar.tick();
    }
    cout<<"loss: "<<loss<<" val_loss: "<<min_val_loss<<endl;

    return {mean, bu, bi, P, Q};
}

int main() {
    string base_path = "/home/nubol23/Desktop/Codes/Cpp/FunkSVD";

    rapidcsv::Document train_csv(base_path + "/train.csv");
    vvi train_data = {train_csv.GetColumn<int>("user_id"), 
                      train_csv.GetColumn<int>("movie_id"),
                      train_csv.GetColumn<int>("rating")};

    rapidcsv::Document val_csv(base_path + "/val.csv");
    vvi val_data = {val_csv.GetColumn<int>("user_id"), 
                    val_csv.GetColumn<int>("movie_id"),
                    val_csv.GetColumn<int>("rating")};

    int n_users = 3974;
    int n_items = 3564;
    int k = 150;
    auto [mu, bu, bi, P, Q] = fit_svd(train_data,
            val_data,
            n_users,
            n_items,
            k,
            0.005,
            0.005,
            0.01,
            0.01,
            0.05,
            0.1,
            75);
    
    return 0;
}