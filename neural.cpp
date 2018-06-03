#include <iostream>
#include <vector>
#include <math.h>

using std::vector;
using std::cout;
using std::endl;

vector<float> X1 {
    5.1, 3.5, 1.4, 0.2,
    4.9, 3.0, 1.4, 0.2,
    6.2, 3.4, 5.4, 2.3,
    5.9, 3.0, 5.1, 1.8
};

vector<float> X2 {
    4.7,3.2,1.3,0.2,
    4.6,3.1,1.5,0.2,
    6.8,3.2,5.9,2.3,
    6.7,3.3,5.7,2.5,
};

vector<float> X3 {
    5.0,3.6,1.4,0.2,
    5.4,3.9,1.7,0.4,
    6.7,3.0,5.2,2.3,
    6.3,2.5,5.0,1.9,
};  

vector<float> y {
    0,
    0,
    1,
    1
};

vector<float> W {
    0.5,
    0.5,
    0.5,
    0.5
};

vector <float> sigmoid_d (const vector <float>& m1) {
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = m1[ i ] * (1 - m1[ i ]);
    }
    
    return output;
}

vector <float> sigmoid (const vector <float>& m1) {

    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + exp(-m1[ i ]));
    }
    
    return output;
}

vector <float> operator+(const vector <float>& m1, const vector <float>& m2){
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

vector <float> operator-(const vector <float>& m1, const vector <float>& m2){
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

vector <float> operator*(const vector <float>& m1, const vector <float>& m2){
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

vector <float> transpose (float *m, const int C, const int R) {
    
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) {

    vector <float> output (m1_rows*m2_columns);
    
    for( int row = 0; row != m1_rows; ++row ) {
        for( int col = 0; col != m2_columns; ++col ) {
            output[ row * m2_columns + col ] = 0.f;
            for( int k = 0; k != m1_columns; ++k ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
    
    return output;
}

void print ( const vector <float>& m, int n_rows, int n_columns ) {

    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

int main(int argc, const char * argv[]) {
    
    for (unsigned i = 0; i != 500000; ++i) {
        
        vector<float> pred = sigmoid(dot(X1, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;        
        vector<float> pred_delta = pred_error * sigmoid_d(pred);        
        vector<float> W_delta = dot(transpose( &X1[0], 4, 4 ), pred_delta, 4, 4, 1);        
        W = W + W_delta;
        
        if (i == 499999){
            print ( pred, 4, 1 );
        };
    };

    for (unsigned i = 0; i != 500000; ++i) {
        
        vector<float> pred = sigmoid(dot(X2, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;        
        vector<float> pred_delta = pred_error * sigmoid_d(pred);        
        vector<float> W_delta = dot(transpose( &X2[0], 4, 4 ), pred_delta, 4, 4, 1);        
        W = W + W_delta;
        
        if (i == 499999){
            print ( pred, 4, 1 );
        };
    };

    for (unsigned i = 0; i != 500000; ++i) {
        
        vector<float> pred = sigmoid(dot(X3, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;        
        vector<float> pred_delta = pred_error * sigmoid_d(pred);        
        vector<float> W_delta = dot(transpose( &X3[0], 4, 4 ), pred_delta, 4, 4, 1);        
        W = W + W_delta;
        
        if (i == 499999){
            print ( pred, 4, 1 );
        };
    };
    return 0;
}
