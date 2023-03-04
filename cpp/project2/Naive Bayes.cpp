#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

const int DATA_SPLIT = 800;

vector<double> calcProbs(const int pclass, const int sex, const double age, const vector<vector<double>> &pclasslh, const vector<vector<double>> &sexlh, const vector<vector<double>> &ageStats, const double priorPos, const double priorNeg);
double calcAgelh(double v, double mean, double var);

int main() {
	fstream fileStream;
	string curLine;
	vector<int> pclass;
	vector<int> survived;
	vector<int> sex;
	vector<double> age;

	//read in data from csv
	fileStream.open("titanic_project.csv", ios::in);
	if (!fileStream.is_open()) return -1; //file open failed, -1 err
	getline(fileStream, curLine); //discard first line
	while (fileStream.is_open() && getline(fileStream, curLine)) {
		curLine = curLine.substr(curLine.find(",", 0) + 1, curLine.length()); //drop first column
		pclass.push_back(stoi(curLine.substr(0, 1)));
		survived.push_back(stoi(curLine.substr(2, 3)));
		sex.push_back(stoi(curLine.substr(4, 5)));
		age.push_back(stod(curLine.substr(6, curLine.length())));
	}
	fileStream.close();

	//calc priors
	double numSurvived = 0;
	double numDied = 0;
	double priorPos = 0;
	double priorNeg = 0;
	for (size_t i = 0; i < DATA_SPLIT; i++) {
		if (survived[i]) numSurvived++;
		else numDied++;
	}
	priorPos = numSurvived / DATA_SPLIT;
	priorNeg = numDied / DATA_SPLIT;
	cout << "Prior Probabilities: Yes=" << priorPos << " , No=" << priorNeg << endl;

	//calc pclass likelihood
	vector<vector<double>> pclasslh(2, vector<double>(3, 0)); //2x3
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			double pcsv = 0;
			for (int k = 0; k < DATA_SPLIT; k++)
				if (pclass[k] == j + 1 && survived[k] == i) pcsv++;
			pclasslh[i][j] = pcsv / (i ? numSurvived : numDied);
		}
	}

	//calc sex likelihood
	vector<vector<double>> sexlh(2, vector<double>(2, 0)); //2x2
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			double sxsv = 0;
			for (int k = 0; k < DATA_SPLIT; k++)
				if (sex[k] == j && survived[k] == i) sxsv++;
			sexlh[i][j] = sxsv / (i ? numSurvived : numDied);
		}
	}

	//calc age mean and variance
	vector<vector<double>> ageStats(2, vector<double>(2, 0)); //2x2 0th row mean, 1st row var
	for (int i = 0; i < 2; i++) {
		double mean = 0;
		for (int j = 0; j < DATA_SPLIT; j++) {
			if (survived[j] == i) {
				mean += age[j];
			}
		}
		mean /= (i ? numSurvived : numDied);
		ageStats[0][i] = mean;
		double var = 0;
		for (int j = 0; j < DATA_SPLIT; j++) {
			if (survived[j] == i) {
				var += pow(age[j] - mean, 2);
			}
		}
		var *= 1.0 / ((i ? numSurvived : numDied)-1);
		ageStats[1][i] = var;
	}

	//predict
	vector<vector<double>> rawProbs;
	for (size_t i = DATA_SPLIT; i < survived.size(); i++) {
		rawProbs.push_back(calcProbs(pclass[i], sex[i], age[i], pclasslh, sexlh, ageStats, priorPos, priorNeg));
	}
	vector<int> predictions;
	for (size_t i = 0; i < rawProbs.size(); i++) {
		predictions.push_back(rawProbs[i][0] > rawProbs[i][1]?1:0);
	}

	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	for (size_t i = 0; i < predictions.size(); i++) {
		if (predictions[i] == 1 && survived[DATA_SPLIT + i] == 1) tp++;
		else if (predictions[i] == 0 && survived[DATA_SPLIT + i] == 0) tn++;
		else if (predictions[i] == 1 && survived[DATA_SPLIT + i] == 0) fp++;
		else if (predictions[i] == 0 && survived[DATA_SPLIT + i] == 1) fn++;
	}
	double accuracy = ((tp + tn) * 1.0) / ((tp + tn + fp + fn) * 1.0);
	double sensitivity = (tp * 1.0) / ((tp + fn) * 1.0);
	double specificity = (tn * 1.0) / ((tn + fp) * 1.0);
	cout << "Accuracy: " << accuracy << "\nSensitivity: " << sensitivity << "\nSpecificity: " << specificity << endl;

	return 0;
}

//function to calculate probability of survival based on given pclass, sex, and age
vector<double> calcProbs(const int pclass, const int sex, const double age, const vector<vector<double>> &pclasslh, const vector<vector<double>> &sexlh, const vector<vector<double>> &ageStats, const double priorPos, const double priorNeg) {
	double numS = pclasslh[1][pclass-1] * sexlh[1][sex] * priorPos * calcAgelh(age, ageStats[0][1], ageStats[1][1]);
	double numD = pclasslh[0][pclass-1] * sexlh[0][sex] * priorNeg * calcAgelh(age, ageStats[0][0], ageStats[1][0]);
	double den = pclasslh[1][pclass-1] * sexlh[1][sex] * calcAgelh(age, ageStats[0][1], ageStats[1][1]) * priorPos + pclasslh[0][pclass-1] * sexlh[0][sex] * calcAgelh(age, ageStats[0][0], ageStats[1][0])* priorNeg;
	return {numS/den, numD/den};
}

//function to calc age likelihood
double calcAgelh(double v, double mean, double var) {
	return 1.0 / sqrt(2.0 * M_PI * var) * exp(-(pow(v - mean, 2) / (2 * var)));
};


