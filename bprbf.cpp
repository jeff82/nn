#include<iostream>
#include<cassert>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<vector>
#include<iomanip>
 
using namespace std;
 
const int P=100;        //输入样本的数量
vector<double> X(P);  //输入样本
vector<double> Y(P);      //输入样本对应的期望输出
const int M=10;         //隐藏层节点数目
vector<double> center(M);       //M个Green函数的数据中心
vector<double> delta(M);        //M个Green函数的扩展常数
double Green[P][M];         //Green矩阵
vector<double> Weight(M);       //权值矩阵
const double eta=0.001;     //学习率
const double ERR=0.9;       //目标误差
const int ITERATION_CEIL=5000;      //最大训练次数
vector<double> error(P);  //单个样本引起的误差
 
/*Hermit多项式函数*/
inline double Hermit(double x){
    return 1.1*(1-x+2*x*x)*exp(-1*x*x/2);
}
 
/*产生指定区间上均匀分布的随机数*/
inline double uniform(double floor,double ceil){
    return floor+1.0*rand()/RAND_MAX*(ceil-floor);
}
 
/*产生区间[floor,ceil]上服从正态分布N[mu,sigma]的随机数*/
inline double RandomNorm(double mu,double sigma,double floor,double ceil){
    double x,prob,y;
    do{
        x=uniform(floor,ceil);
        prob=1/sqrt(2*M_PI*sigma)*exp(-1*(x-mu)*(x-mu)/(2*sigma*sigma));
        y=1.0*rand()/RAND_MAX;
    }while(y>prob);
    return x;
}
 
/*产生输入样本*/
void generateSample(){
    for(int i=0;i<P;++i){
        double in=uniform(-4,4);
        X[i]=in;
        Y[i]=Hermit(in)+RandomNorm(0,0.1,-0.3,0.3);
    }
}
 
/*给向量赋予[floor,ceil]上的随机值*/
void initVector(vector<double> &vec,double floor,double ceil){
    for(int i=0;i<vec.size();++i)
        vec[i]=uniform(floor,ceil);
}
 
/*根据网络，由输入得到输出*/
double getOutput(double x){
    double y=0.0;
    for(int i=0;i<M;++i)
        y+=Weight[i]*exp(-1.0*(x-center[i])*(x-center[i])/(2*delta[i]*delta[i]));

    return y;
}
 
/*计算单个样本引起的误差*/
double calSingleError(int index){
    double output=getOutput(X[index]);
    return Y[index]-output;
}
 
/*计算所有训练样本引起的总误差*/
double calTotalErro(){
    double rect=0.0;
    for(int i=0;i<P;++i){
        error[i]=calSingleError(i);
        rect+=error[i]*error[i];
    }
    return rect/2;
}
 
/*更新网络参数*/
void updateParam(){
    for(int j=0;j<M;++j){
        double delta_center=0.0,delta_delta=0.0,delta_weight=0.0;
        double sum1=0.0,sum2=0.0,sum3=0.0;
        for(int i=0;i<P;++i){
            sum1+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j]);
            sum2+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]))*(X[i]-center[j])*(X[i]-center[j]);
            sum3+=error[i]*exp(-1.0*(X[i]-center[j])*(X[i]-center[j])/(2*delta[j]*delta[j]));
        }
        delta_center=eta*Weight[j]/(delta[j]*delta[j])*sum1;
        delta_delta=eta*Weight[j]/pow(delta[j],3)*sum2;
        delta_weight=eta*sum3;
        center[j]+=delta_center;
        delta[j]+=delta_delta;
        Weight[j]+=delta_weight;
    }
}
 
int main(int argc,char *argv[]){
    srand(time(0));
    /*初始化网络参数*/
    initVector(Weight,-0.1,0.1);
    initVector(center,-4.0,4.0);
    initVector(delta,0.1,0.3);
    /*产生输入样本*/
    generateSample();
    /*开始迭代*/
    int iteration=ITERATION_CEIL;
    while(iteration-->0){
        if(calTotalError()<ERR)      //误差已达到要求，可以退出迭代
            break;
        updateParam();      //更新网络参数
    }
    cout<<"迭代次数:"<<ITERATION_CEIL-iteration-1<<endl;
     
    //根据已训练好的神经网络作几组测试
    for(int x=-4;x<5;++x){
        cout<<x<<"\t";
        cout<<setprecision(8)<<setiosflags(ios::left)<<setw(15);
        cout<<getOutput(x)<<Hermit(x)<<endl;      //先输出我们预测的值，再输出真实值
    }
    return 0;
}