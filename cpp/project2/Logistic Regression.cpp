#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

const int DATA_SPLIT = 800;

bool matrixMult(const vector<vector<double>> &left, const vector<vector<double>> &right, vector<vector<double>> &out);
void sigmoid(vector<vector<double>> &z);

int main() {
	fstream fileStream;
	string curLine;
	vector<double> pclass;
	vector<int> survived;
	vector<double> sex;
	vector<double> age;

	//read in data from csv
	fileStream.open("titanic_project.csv", ios::in);
	if (!fileStream.is_open()) return -1; //file open failed, -1 err
	getline(fileStream, curLine); //discard first line
	while (fileStream.is_open() && getline(fileStream, curLine)) {
		curLine = curLine.substr(curLine.find(",", 0) + 1, curLine.length()); //drop first column
		pclass.push_back(stod(curLine.substr(0, 1)));
		survived.push_back(stoi(curLine.substr(2, 3)));
		sex.push_back(stod(curLine.substr(4, 5)));
		age.push_back(stod(curLine.substr(6, curLine.length())));
	}
	fileStream.close();

	//set up data
	vector<vector<double>> weights = { { 1.0 }, { 1.0 } }; //2x1
	vector<vector<double>> data; //800x2
	vector<vector<double>> dataT = { { }, { } }; //2x800
	vector<vector<double>> testData; //246x2
	for (size_t i = 0; i < sex.size(); i++) {
		if (i < DATA_SPLIT) {
			data.push_back( { 1.0, sex[i] });
			dataT[0].push_back(1.0);
			dataT[1].push_back(sex[i]);
		} else {
			testData.push_back( { 1.0, sex[i] });
		}
	}

	//train
	const double learning_rate = 0.001;
	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < 5000; i++) {
		vector<vector<double>> prob_vec(DATA_SPLIT, vector<double>(1, 0)); //800x1
		vector<vector<double>> error_vec(DATA_SPLIT, vector<double>(1, 0)); //800x1
		vector<vector<double>> error = { { 0 }, { 0 } }; //2x1
		matrixMult(data, weights, prob_vec);
		sigmoid(prob_vec);
		for (int j = 0; j < DATA_SPLIT; j++) error_vec[j][0] = survived[j] - prob_vec[j][0];
		matrixMult(dataT, error_vec, error);
		weights[0][0] = weights[0][0] + (learning_rate * error[0][0]);
		weights[1][0] = weights[1][0] + (learning_rate * error[1][0]);
	}
	auto end = chrono::high_resolution_clock::now();
	auto time = chrono::duration_cast<chrono::milliseconds>(end-start);
	cout << "Training Time: " << time.count() << "ms" << endl;
	cout << "weights = [ " << weights[0][0] << ", " << weights[1][0] << " ]" << endl;

	//predict
	vector<vector<double>> predicted(testData.size(), vector<double>(1, 0)); //nx1
	matrixMult(testData, weights, predicted);
	vector<double> probs(testData.size(), 0);
	for (size_t i = 0; i < testData.size(); i++) {
		probs[i] = exp(predicted[i][0]) / (1 + exp(predicted[i][0]));
	}
	vector<int> predictions(testData.size());
	for (size_t i = 0; i < testData.size(); i++) {
		if (probs[i] > 0.5) predictions[i] = 1;
		else predictions[i] = 0;
	}

	//calculate stats
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

bool matrixMult(const vector<vector<double>> &left, const vector<vector<double>> &right, vector<vector<double>> &out) {
	size_t n = left.size(); //left row size
	size_t m = left[0].size(); //left col size
	size_t o = right.size(); //right row size
	size_t p = right[0].size(); //right col size
	if (m != o) return false; //failed
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < p; j++) {
			for (size_t k = 0; k < m; k++) {
				out[i][j] += left[i][k] * right[k][j];
			}
		}
	}
	return true;
}

void sigmoid(vector<vector<double>> &z) {
	for (size_t i = 0; i < z.size(); i++) {
		for (size_t j = 0; j < z[0].size(); ++j) {
			z[i][j] = 1.0 / (1 + exp(-z[i][j]));
		}
	}
}

