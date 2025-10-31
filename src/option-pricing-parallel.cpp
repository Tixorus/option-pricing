#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>

//https://people.sc.fsu.edu/~jburkardt/cpp_src/toms462/toms462.html
#include "..\include\toms462.hpp"
#include "..\include\toms462.cpp"

struct Stock
{
    double S0;
    double volatility;
};

double calc_real_price(double time, double strike_price, double interest_rate, Stock stock)
{
    double d1 = (log(stock.S0 / strike_price) + ((interest_rate + stock.volatility * stock.volatility / 2) * time)) / (stock.volatility * sqrt(time));
    double d2 = d1 - stock.volatility * sqrt(time);
    double cdfd1 = 0;
    double cdfd2 = 0;
    vdCdfNorm(1, &d1, &cdfd1);
    vdCdfNorm(1, &d2, &cdfd2);
    return stock.S0 * cdfd1 - strike_price * exp(-interest_rate * time) * cdfd2;
}

//https://pure.uva.nl/ws/files/23194494/Thesis.pdf pg 27
//not 100% analytical due to bivariate cdf, but should be very accurate
double calc_real_price_two_no_div(double time, double strike_price, double interest_rate, double rho, Stock stock1, Stock stock2)
{
    double cA = (log(stock1.S0 / strike_price) + (interest_rate + stock1.volatility * stock1.volatility / 2) * time) / (stock1.volatility * sqrt(time));
    double cB = (log(stock2.S0 / strike_price) + (interest_rate + stock2.volatility * stock2.volatility / 2) * time) / (stock2.volatility * sqrt(time));
    double sigma = sqrt(stock2.volatility * stock2.volatility - 2 * rho * stock1.volatility * stock2.volatility + stock1.volatility * stock1.volatility);
    double rhoA = (stock1.volatility - rho * stock2.volatility) / sigma;
    double rhoB = (stock2.volatility - rho * stock1.volatility) / sigma;
    double cHatA = (log(stock1.S0 / stock2.S0) + sigma * sigma * time / 2) / (sigma * sqrt(time));
    double cHatB = (log(stock2.S0 / stock1.S0) + sigma * sigma * time / 2) / (sigma * sqrt(time));
    return stock1.S0 * bivnor(-cHatA, -cA, rhoA) + stock2.S0 * bivnor(-cHatB, -cB, rhoB) - strike_price * exp(-interest_rate * time) * (1 - bivnor(cA - stock1.volatility * sqrt(time), cB - stock2.volatility * sqrt(time), rho));
}


double monte_carlo(bool quasi, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks)
{
    double C = 0;
    int buffer = 65536 / stock_count;

    double timesq = sqrt(time);
    double drift[50], sigma_sqrt[50], s0[50];
    for (unsigned j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * time;
        sigma_sqrt[j] = stocks[j].volatility * timesq;
        s0[j] = stocks[j].S0;
    }

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        int thread_id = omp_get_thread_num();
        double* gauss = new double[buffer * stock_count];
        if (quasi)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths / omp_get_num_threads() + 1) + 1);
        }
        else
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, 123456789);
            vslSkipAheadStream(stream, thread_id * (paths / omp_get_num_threads() + 1) + 1);
        }

#pragma omp for reduction(+:C) schedule(static)
        for (int block = 0; block < paths / buffer; block++)
        {
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
            for (int i = 0; i < buffer; i++)
            {
                double max1 = 0.0, max2 = 0.0;
                for (int j = 0; j < stock_count; j++) 
                {
                    double d = sigma_sqrt[j] * gauss[j + i * stock_count];
                    double e = exp(drift[j] + d);
                    double r = s0[j] * e;
                    if (r > max1) max1 = r;
                    e = exp(drift[j] - d);
                    r = s0[j] * e;
                    if (r > max2) max2 = r;
                }

                C += fmax(0.0, (max1 - strike_price));
                C += fmax(0.0, (max2 - strike_price));
            }
        }
        vslDeleteStream(&stream);
        delete[] gauss;
    }//parallel end

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    //std::cout << "Parallel " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n";
    double ans = (C / paths) * exp(-interest_rate * time) / 2;
    return duration_ms.count();
}

int main()
{
    
    Stock Ar[50];
    
    for (int i = 0; i < 25; i++)
    {
        Stock A;
        Stock B;
        A.S0 = 200;
        B.S0 = 200;
        A.volatility = 0.2;
        B.volatility = 0.3;
        Ar[2*i] = A;
        Ar[2 * i + 1] = B;
    }
    double tim = 0;
    omp_set_num_threads(5);
    for (int i = 0; i < 10; i++)
    {
        tim += monte_carlo(true, 50000000, 0.5, 100, 0.03, 50, Ar);
    }
    cout << tim / 10;
    /*Stock A;
    A.S0 = 200;
    A.volatility = 0.2;
    std::cout << monte_carlo(false, 1000000000, 0.5, 100, 0.03, 1, &A) - calc_real_price(0.5, 100, 0.03, A) << "\n";*/
}