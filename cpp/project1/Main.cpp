#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

double sum(vector<double> &v);
double mean(vector<double> &v);
double median(vector<double> v);
string range(vector<double> v);
double covar(vector<double> &v, vector<double> &w);
double corr(vector<double> &v, vector<double> &w);

int main() {
	vector<double> rm;
	vector<double> medv;
	fstream fileStream;
	string curLine;

	//read in from csv
	fileStream.open("Boston.csv", ios::in);
	if (!fileStream.is_open()) return -1;//file open failed, -1 err
	getline(fileStream, curLine); //discard first line
	while (fileStream.is_open() && getline(fileStream, curLine)) {
		int i = curLine.find(',', 0);
		rm.push_back(stod(curLine.substr(0, i)));
		medv.push_back(stod(curLine.substr(i+1, curLine.length())));
	}
	fileStream.close();

	//compute for rm
	cout << "Compute for rm" << endl;
	cout << "Sum: " << sum(rm) << endl;
	cout << "Mean: " << mean(rm) << endl;
	cout << "Median: " << median(rm) << endl;
	cout << "Range: " << range(rm) << endl << endl;

	//compute for medv
	cout << "Compute for medv" << endl;
	cout << "Sum: " << sum(medv) << endl;
	cout << "Mean: " << mean(medv) << endl;
	cout << "Median: " << median(medv) << endl;
	cout << "Range: " << range(medv) << endl << endl;

	//covar and corr
	cout << "Compute for rm and medv" << endl;
	cout << "Covariance: " << covar(rm, medv) << endl;
	cout << "Correlation: " << corr(rm, medv) << endl;

	return 0;
}

double sum(vector<double> &v) {
	double sum = 0;
	for (const double &i : v) sum += i;
	return sum;
}

double mean(vector<double> &v) {
	double mean = 0;
	for (const double &i : v) mean += i;
	return mean/v.size();
}

double median(vector<double> v) {
	sort(v.begin(),v.end());
	int s = v.size();
	if (s%2==0) return (v[s/2-1] + v[s/2])/2;
	else return v[s/2];
}

string range(vector<double> v) {
	sort(v.begin(),v.end());
	return to_string(v[0])+" "+to_string(v[v.size()-1]);
}

double covar(vector<double> &v, vector<double> &w) {
	double cov = 0;
	double vm = mean(v);
	double wm = mean(w);
	for (size_t i=0;i<v.size();i++) {
		cov += (v[i]-vm)*(w[i]-wm);
	}
	return cov/(v.size()-1);
}

double corr(vector<double> &v, vector<double> &w) {
	double cor = covar(v,w);
	double vvar = 0;
	double wvar = 0;
	double vm = mean(v);
	double wm = mean(w);
	for (const double &x : v) vvar += pow(x-vm,2);
	for (const double &y : w) wvar += pow(y-wm,2);
	vvar = sqrt(vvar/(v.size()-1));
	wvar = sqrt(wvar/(w.size()-1));
	return cor/(vvar*wvar);
}
