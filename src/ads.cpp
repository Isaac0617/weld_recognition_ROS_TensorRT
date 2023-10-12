#include "AdsLib.h"
#include "AdsNotification.h"
#include "AdsVariable.h"

template <typename T>
T read_value(const AdsDevice &route, std::string VAR_NAME){
	AdsVariable<T> Velo{route, VAR_NAME};
	T res = static_cast<T>(Velo);
	return res;
}

double read_value2(const AdsDevice &route, std::string VAR_NAME){
	AdsVariable<double> Velo{route, VAR_NAME};
	double res = static_cast<double>(Velo);
	return res;
}

void write_value(const AdsDevice &route, std::string VAR_NAME, double val){
	AdsVariable<double> target{route, VAR_NAME};
  target = val;
}