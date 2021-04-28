#include<iostream>
#include<Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/LU>
#include "Eigen/Core"
#include "mat.h"
#include <vector>
#include <fstream>
#include <time.h>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>
//#include <algorithm>
//#include <string>
#include "CG_solver.h"
#include <omp.h>

using namespace Eigen;
using namespace std;
using vec = vector<double>;         // vector
using matrix = vector<vec>;            // matrix (=collection of (row) vectors)

int matmatlab(){
	MATFile* pmat;
	pmat = matOpen("Cube3D_tet_N14_April24.mat", "r");
	mxArray* pdata = matGetVariable(pmat,"Nodes");
	return 0;
};

VectorXd CG(MatrixXd& a, VectorXd& b) {			// Vanila conjugate gradient
	int n_b = b.size();
	VectorXd r(n_b), p(n_b), x_CG(n_b);
	x_CG.setZero(); p.setZero(); r.setZero();
	for (int i = 0; i < n_b; i++)
		p[i] = r[i] = b[i];
	int t = 0, T = 200;

	while (t < T)
	{
		double rtr = 0.0;
		double ptAp = 0.0;
		for (int i = 0; i < n_b; i++)
			rtr += r[i] * r[i];
		for (int i = 0; i < n_b; i++)
			for (int j = 0; j < n_b; j++)
				ptAp += a(i,j) * p[i] * p[j];
		double alpha = rtr / (ptAp + 1e-10);
		VectorXd rn(n_b); rn.setZero();
		for (int i = 0; i < n_b; i++)
		{
			x_CG[i] += alpha * p[i];
			rn[i] = r[i];
			for (int j = 0; j < n_b; j++)
				rn[i] -= alpha * a(i,j) * p[j];
		}
		double rntrn = 0.0;
		for (int i = 0; i < n_b; i++)
			rntrn += rn[i] * rn[i];
		if (sqrt(rntrn) < 1e-10) break;
		cout << sqrt(rntrn) << endl;
		double beta = rntrn / rtr;
		for (int i = 0; i < n_b; i++)
		{
			p[i] = beta * p[i] + rn[i];
			r[i] = rn[i];
		}
		t++;
	}
	cout << "\n X_CG \n" << x_CG << "\n" << endl;
	return x_CG;
}

VectorXd GradientDescent(MatrixXd& a, VectorXd& b)
{
	int n_b = b.size();
	MatrixXd  l(n_b, n_b),temp(n_b, n_b);
	VectorXd f(n_b);
	f.setZero(); l.setZero(); temp.setZero();
	int maxIter = 100;
	VectorXd x0(n_b);
	x0.setZero();			//Initial guess
	double tol = 1e-10;
	for (int i = 1; i <= maxIter; i++)
	{
		f = a * x0 - b;
		//cout << " size of l rows:" << f.rows() << " cols: " << f.cols() << endl;
		double rntrn = 0.0;
		rntrn = f.norm();
		if (rntrn < tol)
			return x0;
		temp = f.transpose() * a * f;
		l = f.transpose() * f * temp.inverse();
		//cout << " size of l rows:" << l.rows() <<" cols: "<< l.cols()<<endl;
		//x0 = x0 - l * f;
	}

	return x0;
}




void saveData(string fileName, MatrixXd  matrix)
{
	//https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

MatrixXd localStiff(MatrixXd& CoorNodes) {
	Matrix3d alp1, bet1, gam1, del1, alp2, bet2, gam2, del2, alp3, bet3, gam3, del3, alp4, bet4, gam4, del4;
	alp1 << CoorNodes(0, 1), CoorNodes(1, 1), CoorNodes(2, 1),
		CoorNodes(0, 2), CoorNodes(1, 2), CoorNodes(2, 2),
		CoorNodes(0, 3), CoorNodes(1, 3), CoorNodes(2, 3);
	bet1 << 1, CoorNodes(1, 1), CoorNodes(2, 1),
		1, CoorNodes(1, 2), CoorNodes(2, 2),
		1, CoorNodes(1, 3), CoorNodes(2, 3);
	gam1 << 1, CoorNodes(0, 1), CoorNodes(2, 1),
		1, CoorNodes(0, 2), CoorNodes(2, 2),
		1, CoorNodes(0, 3), CoorNodes(2, 3);
	del1 << 1, CoorNodes(0, 1), CoorNodes(1, 1),
		1, CoorNodes(0, 2), CoorNodes(1, 2),
		1, CoorNodes(0, 3), CoorNodes(1, 3);
	alp2 << CoorNodes(0, 0), CoorNodes(1, 0), CoorNodes(2, 0),
		CoorNodes(0, 2), CoorNodes(1, 2), CoorNodes(2, 2),
		CoorNodes(0, 3), CoorNodes(1, 3), CoorNodes(2, 3);
	bet2 << 1, CoorNodes(1, 0), CoorNodes(2, 0),
		1, CoorNodes(1, 2), CoorNodes(2, 2),
		1, CoorNodes(1, 3), CoorNodes(2, 3);
	gam2 << 1, CoorNodes(0, 0), CoorNodes(2, 0),
		1, CoorNodes(0, 2), CoorNodes(2, 2),
		1, CoorNodes(0, 3), CoorNodes(2, 3);
	del2 << 1, CoorNodes(0, 0), CoorNodes(1, 0),
		1, CoorNodes(0, 2), CoorNodes(1, 2),
		1, CoorNodes(0, 3), CoorNodes(1, 3);
	alp3 << CoorNodes(0, 0), CoorNodes(1, 0), CoorNodes(2, 0),
		CoorNodes(0, 1), CoorNodes(1, 1), CoorNodes(2, 1),
		CoorNodes(0, 3), CoorNodes(1, 3), CoorNodes(2, 3);
	bet3 << 1, CoorNodes(1, 0), CoorNodes(2, 0),
		1, CoorNodes(1, 1), CoorNodes(2, 1),
		1, CoorNodes(1, 3), CoorNodes(2, 3);
	gam3 << 1, CoorNodes(0, 0), CoorNodes(2, 0),
		1, CoorNodes(0, 1), CoorNodes(2, 1),
		1, CoorNodes(0, 3), CoorNodes(2, 3);
	del3 << 1, CoorNodes(0, 0), CoorNodes(1, 0),
		1, CoorNodes(0, 1), CoorNodes(1, 1),
		1, CoorNodes(0, 3), CoorNodes(1, 3);
	alp4 << CoorNodes(0, 0), CoorNodes(1, 0), CoorNodes(2, 0),
		CoorNodes(0, 1), CoorNodes(1, 1), CoorNodes(2, 1),
		CoorNodes(0, 2), CoorNodes(1, 2), CoorNodes(2, 2);
	bet4 << 1, CoorNodes(1, 0), CoorNodes(2, 0),
		1, CoorNodes(1, 1), CoorNodes(2, 1),
		1, CoorNodes(1, 2), CoorNodes(2, 2);
	gam4 << 1, CoorNodes(0, 0), CoorNodes(2, 0),
		1, CoorNodes(0, 1), CoorNodes(2, 1),
		1, CoorNodes(0, 2), CoorNodes(2, 2);
	del4 << 1, CoorNodes(0, 0), CoorNodes(1, 0),
		1, CoorNodes(0, 1), CoorNodes(1, 1),
		1, CoorNodes(0, 2), CoorNodes(1, 2);
	double det_alp1, det_bet1, det_gam1, det_del1, det_alp2, det_bet2, det_gam2, det_del2, det_alp3, det_bet3, det_gam3,
		det_del3, det_alp4, det_bet4, det_gam4, det_del4;
	det_alp1 = alp1.determinant();
	det_bet1 = -bet1.determinant();
	det_gam1 = gam1.determinant();
	det_del1 = -del1.determinant();
	det_alp2 = -alp2.determinant();
	det_bet2 = bet2.determinant();
	det_gam2 = -gam2.determinant();
	det_del2 = del2.determinant();
	det_alp3 = alp3.determinant();
	det_bet3 = -bet3.determinant();
	det_gam3 = gam3.determinant();
	det_del3 = -del3.determinant();
	det_alp4 = -alp4.determinant();
	det_bet4 = bet4.determinant();
	det_gam4 = -gam4.determinant();
	det_del4 = del4.determinant();
	cout << "The det_alp1 is: " << det_alp1 << det_del4;
	double V;
	Matrix4d V6;
	V6 << 1, CoorNodes(0, 0), CoorNodes(1, 0), CoorNodes(2, 0),
		1, CoorNodes(0, 1), CoorNodes(1, 1), CoorNodes(2, 1),
		1, CoorNodes(0, 2), CoorNodes(1, 2), CoorNodes(2, 2),
		1, CoorNodes(0, 3), CoorNodes(1, 3), CoorNodes(2, 3);
	V = 1.0 / 6 *V6.determinant();

	// cout << "Determinant of V : " << det_V << "\n";

	MatrixXd B1(6, 3), B2(6, 3), B3(6, 3), B4(6, 3), B(6, 12);
	B1 << det_bet1, 0, 0,
		0, det_gam1, 0,
		0, 0, det_del1,
		det_gam1, det_bet1, 0,
		0, det_del1, det_gam1,
		det_del1, 0, det_bet1;
	B2 << det_bet2, 0, 0,
		0, det_gam2, 0,
		0, 0, det_del2,
		det_gam2, det_bet2, 0,
		0, det_del2, det_gam2,
		det_del2, 0, det_bet2;
	B3 << det_bet3, 0, 0,
		0, det_gam3, 0,
		0, 0, det_del3,
		det_gam3, det_bet3, 0,
		0, det_del3, det_gam3,
		det_del3, 0, det_bet3;
	B4 << det_bet4, 0, 0,
		0, det_gam4, 0,
		0, 0, det_del4,
		det_gam4, det_bet4, 0,
		0, det_del4, det_gam4,
		det_del4, 0, det_bet4;

	B << B1, B2, B3, B4;
	cout << "The B matrix is: " << B << "\n";
	MatrixXd D(6, 6), k(12, 12);
	double nu = 0.3, E = 210*pow(10, 9);				// Material Properties of Steel
	D << 1 - nu, nu, nu, 0, 0, 0,
		nu, 1 - nu, nu, 0, 0, 0,
		nu, nu, 1 - nu, 0, 0, 0,
		0, 0, 0, (1 - 2 * nu) / 2, 0, 0,
		0, 0, 0, 0, (1 - 2 * nu) / 2, 0,
		0, 0, 0, 0, 0, (1 - 2 * nu) / 2;
	//cout << "The D matrix before scaler multiplication is: " << D << "\n";
	D = E / ((1 + nu) * (1 - 2 * nu)) * D;
	//cout << "The D matrix is: " << D << "\n";
	k = B.transpose() * D * B * V;
	//cout << "The element stiffness k is : " << k << "\n";
	Eigen::LLT<Eigen::MatrixXd> lltOfk(k); // compute the Cholesky decomposition of A
	if (!k.isApprox(k.transpose()) ) // || lltOfk.info() == Eigen::NumericalIssue
	{
		printf("Possibly not a symmetric positive definitie matrix!");
	}
	return k;
}

int main() {

	MatrixXd m = MatrixXd::Random(3, 3);
	int a[5] = { 2,5,6,5,9 };
	m = (m + MatrixXd::Constant(3, 3, 1.2)) * 50;
	cout << "m = " << endl << m << endl;
	VectorXd v(3);
	v << 5, 2, 3;
	cout << "m * v = " << endl << m * v << endl;

	// read data from text file
	double *Nodes, *Elements, *Nodes_tf2, *Nodes_tf6;
	MATFile* pmatFile = NULL;
	mxArray* pMxArray = NULL;
	pmatFile = matOpen("Cube3D_tet_N14_April24.mat", "r");
	pMxArray = matGetVariable(pmatFile, "Nodes");
	Nodes = (double*)mxGetData(pMxArray);
	int M, N;
	M = mxGetM(pMxArray);
	N = mxGetN(pMxArray);
	int nNodes = N;		// number of nodes
	//Matrix<double> A(M, N);
	MatrixXd  A(M, N);
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			A(i,j) = Nodes[M * j + i];
	cout <<" Nodes A: \n" <<A << " \n";
	mxFree(Nodes);		// Cleaning
	// Read elements 
	mxArray* qMxArray = NULL;
	qMxArray = matGetVariable(pmatFile, "Elements");
	Elements = (double*)mxGetData(qMxArray);
	int MM, NN;
	MM = mxGetM(qMxArray);
	NN = mxGetN(qMxArray);
	int nElm = NN;				// number of elements
	MatrixXd  EE(MM, NN);
	for (int i = 0; i < MM; i++)
		for (int j = 0; j < NN; j++)
			EE(i, j) = Elements[MM * j + i];
	cout << "elements : \n"<<EE << "\n ";
	mxFree(Elements);		// Cleaning
	mxArray* rMxArray = NULL;				// Read the bottom face nodes 
	rMxArray = matGetVariable(pmatFile, "Nodes_f2");
	Nodes_tf2 = (double*)mxGetData(rMxArray);
	int M3, N3;
	M3 = mxGetM(rMxArray);
	N3 = mxGetN(rMxArray);
	int nNodesF2 = N3;				// number of elements
	VectorXd  constraint_nodes(N3);
	for (int j = 0; j < N3; j++)
		constraint_nodes(j) = Nodes_tf2[j];
	mxArray* sMxArray = NULL;				// Read the top face nodes 
	sMxArray = matGetVariable(pmatFile, "Nodes_f6");
	Nodes_tf6 = (double*)mxGetData(sMxArray);
	int M4, N4;
	M4 = mxGetM(sMxArray);
	N4 = mxGetN(sMxArray);
	int nNodesF6 = N4;				// number of elements
	VectorXd  force_nodes(N4);
	for (int j = 0; j < N4; j++)
		force_nodes(j) = Nodes_tf6[j];
	//VectorXd Nod_f6(1, N4);
	//Nod_f6 = Nodes_f6(seq(1, 1), seq(1, N4));
	matClose(pmatFile);		// Close the input file		

	// Assembly of the element stiffness matrix
	MatrixXd  globalStiff(3*nNodes, 3*nNodes);				//  Hold global stiffness matrix
	globalStiff.setZero(3 * nNodes, 3 * nNodes);

	int fI, sI, tI, rI;									// indices
	for (int j = 0; j < nElm; j++) {					// loop over elements i.e nElm
		VectorXd elmNodes(4);
		elmNodes  = EE(seq(0, 3), j);
		
		MatrixXd  temp_elmNodes(3, 4);
		int nn;
		for (int i = 0; i < 4; i++) {
			nn = elmNodes(i, 0);
			cout << "\n ele row: \n" << EE(i, j) << " nn "<< nn << "\n ";
			temp_elmNodes.col(i) = A.col((nn - 1));
		}
		cout << "\n temp elmNodes :\n"<< j <<" \n " << temp_elmNodes << "\n";

		MatrixXd temp_localStiff(12, 12);					// To store the returned matrix.
		temp_localStiff.setZero(12, 12);					// initialization
		temp_localStiff = localStiff(temp_elmNodes);
		//cout << "\n element stiffness :\n" << j << " \n " << temp_localStiff << "\n";
		for (int i = 0; i < nNodes; i++) {				// loop over nodes of each element

			fI = ((EE(0, j) - 1) * 3);
			sI = ((EE(1, j) - 1) * 3);
			tI = ((EE(2, j) - 1) * 3);
			rI = ((EE(3, j) - 1) * 3);
			
			globalStiff(seq(fI, fI + 2), seq(fI, fI + 2)) += temp_localStiff(seq(0, 2), seq(0, 2));
			globalStiff(seq(fI, fI + 2), seq(sI, sI + 2)) += temp_localStiff(seq(0, 2), seq(3, 5));
			globalStiff(seq(fI, fI + 2), seq(tI, tI + 2)) += temp_localStiff(seq(0, 2), seq(6, 8));
			globalStiff(seq(fI, fI + 2), seq(rI, rI + 2)) += temp_localStiff(seq(0, 2), seq(9, 11));
			
			globalStiff(seq(sI, sI + 2), seq(fI, fI + 2)) += temp_localStiff(seq(3, 5), seq(0, 2));
			globalStiff(seq(sI, sI + 2), seq(sI, sI + 2)) += temp_localStiff(seq(3, 5), seq(3, 5));
			globalStiff(seq(sI, sI + 2), seq(tI, tI + 2)) += temp_localStiff(seq(3, 5), seq(6, 8));
			globalStiff(seq(sI, sI + 2), seq(rI, rI + 2)) += temp_localStiff(seq(3, 5), seq(9,11));
			
			globalStiff(seq(tI, tI + 2), seq(fI, fI + 2)) += temp_localStiff(seq(6, 8), seq(0, 2));
			globalStiff(seq(tI, tI + 2), seq(sI, sI + 2)) += temp_localStiff(seq(6, 8), seq(3, 5));
			globalStiff(seq(tI, tI + 2), seq(tI, tI + 2)) += temp_localStiff(seq(6, 8), seq(6, 8));
			globalStiff(seq(tI, tI + 2), seq(rI, rI + 2)) += temp_localStiff(seq(6, 8), seq(9, 11));
			
			globalStiff(seq(rI, rI + 2), seq(fI, fI + 2)) += temp_localStiff(seq(9, 11), seq(0, 2));
			globalStiff(seq(rI, rI + 2), seq(sI, sI + 2)) += temp_localStiff(seq(9, 11), seq(3, 5));
			globalStiff(seq(rI, rI + 2), seq(tI, tI + 2)) += temp_localStiff(seq(9, 11), seq(6, 8));
			globalStiff(seq(rI, rI + 2), seq(rI, rI + 2)) += temp_localStiff(seq(9, 11), seq(9, 11));
		}
	}
	// Check if global stiffness is symmetric.
	if (!globalStiff.isApprox(globalStiff.transpose()))
	{
		printf("Possibly global stiffness matrix is not a symmetric matrix!");
	}
	else {
		printf("The Global stiffness matrix before BC is symmetric");
	}

	// setting constraints (on the bottom of the cube)
	// % face 6 is the top face (for force); face 2 is the bottom face (for constraint)
	//VectorXd constraint_nodes(nNodesF2), force_nodes(nNodesF6);
	VectorXd forceVec(3 * N);
	forceVec.setZero();
	int BC =0, dim = 3, BI;						// BC-> number of constraint nodes
	//constraint_nodes << 2, 3, 5, 6, 10;			// constraints nodes on the bottom surface
	//force_nodes << 1, 4, 7, 8, 14;				// force on all the nodes of the top surface


	// apply force on the top surface nodes
	VectorXd f_nodes_temp;				// constant force on each of the nodes in 
	int temp = 0;
	for (int i = 0; i < force_nodes.size(); i++) {	// iterate over the nodes on which force is applied
		temp = force_nodes(i);
		temp = temp * 3 - 1;						// The axis on which the force is applied.
		forceVec[temp] = -15000000000;
	}
	f_nodes_temp = forceVec;


	BC = constraint_nodes.size();
	for (int i = 0; i < BC; i++) {

		BI = ((constraint_nodes(i)-1) * 3);
		MatrixXd Z = MatrixXd::Zero(3,dim);
		globalStiff.block(BI, 0, 3, dim).setZero(); // = Z;
		//cout << "\n global stiffness zeros 1st \n" << globalStiff.block(BI, 1, 3, dim) << " \n";

		globalStiff.block(0, BI, dim, 3).setZero();// = Z;
		//cout << "\n global stiffness zeros  2nd \n" << globalStiff.block(1, BI, dim, 3) << " \n";

		MatrixXd I = MatrixXd::Identity(1, 1);
		globalStiff.block(BI, BI, 1, 1) = I;
		//cout << "\n global stiffness Identity  3nd \n" << globalStiff.block(BI, BI, 1, 1) << " \n";
		int temp = BI + 1;
		globalStiff.block(temp,temp, 1, 1).setIdentity();	// = I;
		//cout << "\n global stiffness Identity  4th \n" << globalStiff.block((BI + 1), (BI + 1), 1, 1) << " \n";
		int temp2 = BI + 2;
		globalStiff.block(temp2, temp2, 1, 1).setIdentity();
		//cout << "\n global stiffness Identity  5th \n" << globalStiff.block((BI + 2), (BI + 2), 1, 1) << " \n";

		VectorXd fv = VectorXd::Zero(3);	// segment of force vector to be zero
		// forceVec(seq(BI, BI + 2)) = fv;

	}
	cout << "\n global stiff after BC: \n" << globalStiff << " ";
	Eigen::LLT<Eigen::MatrixXd> lltOfglobalStiff(globalStiff); // compute the Cholesky decomposition of A
	if (!globalStiff.isApprox(globalStiff.transpose()) || lltOfglobalStiff.info() == Eigen::NumericalIssue)
	{
		printf("Possibly not a symmetric positive definitie matrix!");
	}

	if (!globalStiff.isApprox(globalStiff.transpose()) )
	{
		printf("global stiffness matrix after BC is NOT a symmetric matrix!");
	}
	VectorXd x(3 * nNodes), xx(3 * nNodes);
	auto start = chrono::steady_clock::now();

	//omp_set_num_threads(4);
	//#pragma omp parallel
	//{
	VectorXd err(200);
		int ID = omp_get_thread_num();
		for (int i = 0; i < 200; i++) {
			x.setZero();
			ConjugateGradient<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, Lower | Upper> cg;
			cg.compute(globalStiff);
			cg.setTolerance(0.01);
			cg.setMaxIterations((i+1));
			x = cg.solve(forceVec);
			err(i) = cg.error();
			cout << "\n #iterations:     " << cg.iterations() <<"\n" <<endl;
			cout << "\n estimated error: " << cg.error() << "\n" << endl;
		}
		printf("the id is %d", ID);
		cout << "\n #err :     " << err << "\n" << endl;

	//}
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << chrono::duration <double, milli>(diff).count() << " ms \n" << endl;


	VectorXd x1_CG(3 * nNodes);
	x1_CG = CG(globalStiff, forceVec);

	VectorXd x2_GD(3 * nNodes);
	x2_GD = GradientDescent(globalStiff, forceVec);
	//cout << "The least-squares solution is:\n"
		//<< globalStiff.bdcSvd(ComputeThinU | ComputeThinV).solve(forceVec) << endl;
	
	// BiCGSTAB<SparseMatrix<double> > solver;

	/*int nrow = 3 * nNodes, ncol = 3 * nNodes;
	int arr2[42][42];
	Map<MatrixXd>(&arr2[0][0], nrow, ncol) = globalStiff;
	double* resultC;        // NULL pointer <-- WRONG INFO from the site. resultC must be preallocated!
	Map<MatrixXd>(resultC, globalStiff.rows(), globalStiff.cols()) = globalStiff;

	double* vc = globalStiff.data();
	double* B = forceVec.data();
	vec X = conjugateGradientSolver(vc, B);
	
	/*omp_set_num_threads(4);
	#pragma omp parallel
	{
		int ID = omp_get_thread_num();
		printf("Hello %d", ID);
		printf("world %d \n", ID);
	}*/

	//saveData("matrix_N880.csv", globalStiff);
	//saveData("x_N880.csv", x);
  	return 0;
}