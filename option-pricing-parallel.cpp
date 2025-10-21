/* TODO:
- interface
- optimisation (buffer for random numbers?)
- control variates
- dividends not 0 (dunno if you can make that work analytically for n = 2)
- correlation not 0 (for monte-carlo)
- american (Longstaff-Schwartz?), bermuda options
- non constant dividents, volatility, interest rate
*/


#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>

//https://people.sc.fsu.edu/~jburkardt/cpp_src/toms462/toms462.html
#include "F:\cppstuff\toms\toms462.hpp"

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


double monte_carlo(bool quasi, int paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks)
{
    auto start = std::chrono::high_resolution_clock::now();
    double C = 0;

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        int thread_id = omp_get_thread_num();

        double* gauss = new double[stock_count];
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
        for (int i = 0; i < paths; i++)
        {
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, stock_count, gauss, 0.0, 1.0);
            double max_return1 = 0;
            double max_return2 = 0;
            for (int j = 0; j < stock_count; j++)
            {
                double mult1 = exp((interest_rate - stocks[j].volatility * stocks[j].volatility / 2) * time + stocks[j].volatility * sqrt(time) * gauss[j]);
                double mult2 = exp((interest_rate - stocks[j].volatility * stocks[j].volatility / 2) * time - stocks[j].volatility * sqrt(time) * gauss[j]);
                max_return1 = fmax(max_return1, stocks[j].S0 * mult1);
                max_return2 = fmax(max_return2, stocks[j].S0 * mult2);
            }
            C += fmax(0, (max_return1 - strike_price) / 2);
            C += fmax(0, (max_return2 - strike_price) / 2);
        }
        
        vslDeleteStream(&stream);
        delete[] gauss;
    }//parallel end

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << "Parallel " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n";

    return (C / paths) * exp(-interest_rate * time);
}

int main()
{
    omp_set_num_threads(6);
    std::cout << std::scientific;
    /*
    Stock A;
    Stock B;
    A.S0 = 200;
    B.S0 = 200;
    A.volatility = 0.2;
    B.volatility = 0.3;
    Stock Ar[2] = {A, B};
    std::cout << calc_real_price_two_no_div(0.5, 100, 0.03, 0, A, B)- monte_carlo_quasi(105000000, 0.5, 100, 0.03, 2, Ar) << "\n";*/
    Stock A;
    A.S0 = 200;
    A.volatility = 0.2;
    std::cout << monte_carlo(false, 100000000, 0.5, 100, 0.03, 1, &A) - calc_real_price(0.5, 100, 0.03, A) << "\n";

}
