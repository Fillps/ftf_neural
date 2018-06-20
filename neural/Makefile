# C compiler
CC = g++ -std=c++11

destDir = /tmp/neural

all: gen_bin gen_data mv_bin

gen_bin: neural.cpp
	$(CC) neural.cpp -o neural

gen_data: gen_bin
	./neural input.txt > gold

mv_bin: gen_data
	mkdir -p $(destDir)
	mv neural gold $(destDir)
	cp input.txt $(destDir)
	@echo "All Done, executables should be in the $(destDir)"


