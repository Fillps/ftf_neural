#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>

using std::vector;
using std::cout;
using std::endl;

const char* detectLog;

unsigned int linesCount(std::string s)
{
    unsigned int count = 0;
    int posMax = s.length();

    for (int position = 0; position < posMax; )
        if (s[position++] == '\n')
            count++;

    return count;
}

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
    vector<float> output(VECTOR_SIZE);

	
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        output[i] = 1 / (1 + expf(-m1[ i ]));
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

void print ( const vector <float>& m, int n_rows, int n_columns, std::ostringstream* stream) {

    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            *stream << m[ i * n_columns + j ] << " ";
        }
        *stream << '\n';
    }
    *stream << endl;
}

void load_matrix( const char* filename, vector<float> *X1, vector<float> *X2, vector<float> *X3, vector<float> *y, vector<float> *W){
    std::ifstream in(filename);

    if (!in) {
		std::ofstream err_detected(detectLog);
        if (err_detected) {
            err_detected << "No input file.\n";
            err_detected.close();
        }
        exit(-1);
    }

    for (int i = 0; i < 16; i++)
        in >> (*X1)[i];
    for (int i = 0; i < 16; i++)
        in >> (*X2)[i];
    for (int i = 0; i < 16; i++)
        in >> (*X3)[i];
    for (int i = 0; i < 4; i++)
        in >> (*y)[i];
    for (int i = 0; i < 4; i++)
        in >> (*W)[i];

    in.close();
}

int neural(const char * input_name, std::ostringstream* stream) {

    vector<float> X1(16), X2(16), X3(16), y (4), W(4);

    load_matrix(input_name, &X1, &X2, &X3, &y, &W);


    for (unsigned i = 0; i != 500000; ++i) {

        vector<float> pred = sigmoid(dot(X1, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;
        vector<float> pred_delta = pred_error * sigmoid_d(pred);
        vector<float> W_delta = dot(transpose( &X1[0], 4, 4 ), pred_delta, 4, 4, 1);
        W = W + W_delta;

        if (i == 499999){
            print ( pred, 4, 1, stream );
        };
    };

    for (unsigned i = 0; i != 500000; ++i) {

        vector<float> pred = sigmoid(dot(X2, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;
        vector<float> pred_delta = pred_error * sigmoid_d(pred);
        vector<float> W_delta = dot(transpose( &X2[0], 4, 4 ), pred_delta, 4, 4, 1);
        W = W + W_delta;

        if (i == 499999){
            print ( pred, 4, 1, stream );
        };
    };

    for (unsigned i = 0; i != 500000; ++i) {

        vector<float> pred = sigmoid(dot(X3, W, 4, 4, 1 ) );
        vector<float> pred_error = y - pred;
        vector<float> pred_delta = pred_error * sigmoid_d(pred);
        vector<float> W_delta = dot(transpose( &X3[0], 4, 4 ), pred_delta, 4, 4, 1);
        W = W + W_delta;

        if (i == 499999){
            print ( pred, 4, 1, stream );
        };
    };
    return 0;
}

int main(int argc, const char * argv[]){
    if (argc == 4){
        detectLog = argv[2];  
    } else {
        fprintf(stderr, "Usage: %s <input file> <detectLog> <output file>\n", argv[0]);
        exit(-1);
    }

    std::ostringstream stream;
    neural( argv[1], &stream );

    if (linesCount(stream.str()) != 15){
        std::ofstream err_detected(argv[2]);
        if (err_detected) {
            err_detected << "Wrong number of lines: " << linesCount(stream.str()) << endl << stream.str();
            err_detected.close();
        }
    }

    std::ofstream out(argv[3]);
    if (out){
        out << stream.str();
        out.close();
    }
}
