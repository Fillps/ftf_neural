# C compiler
CC = g++ -std=c++11

destDir = /tmp/neural-FullDuplication-Threads

all: gen_bin gen_data mv_bin

gen_bin: neural-fulldup-threads.cpp
	$(CC) neural-fulldup-threads.cpp -o neural-fulldup-threads

gen_data: gen_bin
	./neural-fulldup-threads input.txt detection.log > gold

mv_bin: gen_data
	mkdir -p $(destDir)
	mv neural-fulldup-threads gold input.txt $(destDir)
	@echo "All Done, executables should be in the $(destDir)"


