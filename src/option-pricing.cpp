#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"

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
    double sigma = sqrt(stock2.volatility * stock2.volatility - 2*rho*stock1.volatility*stock2.volatility + stock1.volatility * stock1.volatility);
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
    int buffer = 50000;

    double timesq = sqrt(time);
    double discount = exp(-interest_rate * time);
    double drift, sigma_sqrt;
    Stock A = stocks[0];
    drift = (interest_rate - 0.5 * A.volatility * A.volatility) * time;
    sigma_sqrt = A.volatility * timesq;
    VSLStreamStatePtr stream;
    double* gauss = new double[buffer * stock_count];
    if (quasi)
    {
        vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
        vslSkipAheadStream(stream, 1);
    }
    else
    {
        vslNewStream(&stream, VSL_BRNG_MCG59, 123456789);
        vslSkipAheadStream(stream, 1);
    }

    for (int block = 0; block < paths / buffer; block++)
    {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
        for (int i = 0; i < buffer; i++)
        {
            double max1 = 0.0, max2 = 0.0;
            double d = sigma_sqrt * gauss[0 + i * stock_count];
            double e1 = exp(drift + d);
            double e2 = exp(drift - d);
            double r1 = A.S0 * e1;
            double r2 = A.S0 * e2;
            C += fmax(0.0, (r1 - strike_price));
            C += fmax(0.0, (r2 - strike_price));
        }
    }
    vslDeleteStream(&stream);
    delete[] gauss;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << "Sequential " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n";

    return (C / paths) * exp(-interest_rate * time) / 2;
}

int main()
{
    std::cout << std::scientific;

    Stock A;
    Stock B;
    A.S0 = 200;
    B.S0 = 200;
    A.volatility = 0.2;
    B.volatility = 0.3;
    Stock Ar[2] = { A, B };
    std::cout << calc_real_price_two_no_div(0.5, 100, 0.03, 0, A, B) - monte_carlo(true, 1000000000, 0.5, 100, 0.03, 1, Ar) << "\n";

}
