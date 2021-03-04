#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string.h>

#define MAX_TRAIN 27455
#define MAX_TESTS 7172
#define Pixels 784
using namespace std;

double logSigmoid(double x, int a){
    return (double)1/((double)1+exp(double(-1*a*x)));
}

int main()
{   srand (0);

    string line;
    int value;
    ifstream myfile ("sign_mnist_train.csv");
    if (myfile.is_open())
    {
        cout<<"the data file is found"<<endl;
    }
    else cout << "Unable to open file";

    int L,M,N;
//    cout<<"please enter no of input neurons"<<endl;
//    cin>>L;
L = Pixels;

    N = 26;
    M = 100;

    int nTrain,nTest;
//    cout<<"please enter no of training patterns"<<endl;
//    cin>>nTrain;
//    cout<<"please enter no of testing patterns"<<endl;
//    cin>>nTest;
nTrain = MAX_TRAIN;
nTest = MAX_TESTS;

    double v[M][L];
    for (int j=0;j<M;j++){
        for (int i=0;i<L;i++){
            v[j][i]=((double)(rand()%500))/((double)500);
            //cout<<v[i][j]<<" ";
        }
        //cout<<endl;
    }

    double w[N][M+1];
    for (int k=0;k<N;k++){
        for (int j=0;j<M+1;j++){
            w[k][j]=((double)(rand()%500))/((double)500);
            //cout<<w[i][j]<<" ";
        }
        //cout<<endl;
    }

    double hidden[M+1]={0};
    double output[N]={0};
    double learningRate=0.2;

    //initializing the bias neuron
    hidden[M]=((double)rand())/((double)RAND_MAX);

    double inputs[L];

//here comes the loop
    for (int p=0;p<nTrain;p++){
    double targets[N]={0};
    getline (myfile,line,',');
    value = stoi(line);
    targets[value]=1;
    //cout<<targets[value]<<"target"<<line<<endl;
    int x=1;
    while ( x<784 )
    {getline (myfile,line,',');
        value = stoi(line);
        inputs[x-1]=(double)value/double(256.0);
        x++;
    }
    getline (myfile,line,'\n');
    value = stoi(line);
    inputs[x-1]=(double)value/double(256.0);
    //cout<<inputs[0]<<" "<<inputs[L-1]<<endl;




//forward_pass();
    //coding for one pass
    //for hidden layer values

    for (int j=0;j<M;j++){
        hidden[j]=0;
        for (int i=0;i<L;i++){
            hidden[j]+=v[j][i]*inputs[i];
            //cout<<hidden[j]<<"hidden"<<endl;
        }
        hidden[j]/=784;

        hidden[j]=logSigmoid(hidden[j],1);
        //cout<<hidden[j]<<" den";

        //cout<<endl;

    }

    //for output layer values
    for (int k=0;k<N;k++){
        output[k]=0;
        for (int j=0;j<M+1;j++){
            output[k]+=w[k][j]*hidden[j];
        }
        output[k]/=100;
        output[k]=logSigmoid(output[k],1);
        //cout<<output[k]<<"output"<<endl;
    }

//    for (int i=0;i<N;i++){
//        cout<<output[i]<<" ";
//    }
//    cout<<endl;
//
//    for (int i=0;i<L;i++){
//        cout<<inputs[i]<<" ";
//    }
//    cout<<endl<<endl;

    //forward propagation done

    //code for back propagation


    for (int k=0;k<N;k++){
        for (int j=0;j<M+1;j++){
            w[k][j]+=learningRate*(targets[k]-output[k])*1.0*output[k]*(1-output[k])*hidden[j];
        }
    }



    for (int j=0;j<M;j++){
        for (int i=0;i<L;i++){
            double error=0;
            for (int k=0;k<N;k++){
                error+=learningRate*(targets[k]-output[k])*1.0*output[k]*(1-output[k])*w[k][j]*1.0*hidden[j]*(1-hidden[j])*inputs[i];
            }
            v[j][i]=error/(double)N;
        }
    }

    }



    ifstream mfile ("sign_mnist_test.csv");
cout<<"testing"<<endl;
//testing the model
    for (int p=0;p<nTest;p++){
    double targets[N]={0};
    getline (mfile,line,',');
    value = stoi(line);
    targets[value]=1;
    int x=1;
    while ( x<784 )
    {getline (mfile,line,',');
        value = stoi(line);
        inputs[x-1]=(double)value/double(256.0);
        x++;
    }
    getline (mfile,line,'\n');
    value = stoi(line);
    inputs[x-1]=(double)value/double(256.0);

    for (int j=0;j<M;j++){
        hidden[j]=0;
        for (int i=0;i<L;i++){
            hidden[j]+=v[j][i]*inputs[i];
            //cout<<hidden[j]<<"hidden"<<endl;
        }
        hidden[j]/=784;

        hidden[j]=logSigmoid(hidden[j],1);
        //cout<<hidden[j]<<" den";

        //cout<<endl;

    }

    //for output layer values
    for (int k=0;k<N;k++){
        output[k]=0;
        for (int j=0;j<M+1;j++){
            output[k]+=w[k][j]*hidden[j];
        }
        output[k]/=100;
        output[k]=logSigmoid(output[k],1);
        //cout<<output[k]<<"output"<<endl;
    }

    for (int i=0;i<N;i++){
        cout<<output[i]<<" ";
    }
    cout<<endl;

    for (int i=0;i<N;i++){
        cout<<targets[i]<<" ";
    }
    cout<<endl<<endl;
    }




    myfile.close();
    mfile.close();
    return 0;
}
