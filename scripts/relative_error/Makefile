# numpy, R and ggplot must be installed. For ubuntu users: sudo pip install numpy | sudo apt install r-base-core r-cran-ggplot2
DIR?=../../logs
NAME?=neural
all: neural_output_parser.py parseUtils.py createErrLimitsTableforPlot.R lineGraph.R 
	chmod 777 neural_output_parser.py
	./neural_output_parser.py ${DIR}
	chmod 777 process.py
	./process.py
	./createErrLimitsTableforPlot.R out.csv ${NAME} errLimit.csv
	./lineGraph.R errLimit.csv errLimit.pdf

clear:
	rm -f out.csv errLimit.csv errLimit.pdf parseUtils.pyc process.py
