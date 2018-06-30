# C compiler
CC = g++ -std=c++11

destDir = /tmp/neural-tmr

all: gen_bin gen_data mv_bin

gen_bin: neural-tmr.cpp
	$(CC) neural-tmr.cpp -o neural-tmr

gen_data: gen_bin
	./neural-tmr input.txt detection.log gold

mv_bin: gen_data
	mkdir -p $(destDir)
	mv neural-tmr gold $(destDir)
	cp input.txt $(destDir)
	@echo "All Done, executables should be in the $(destDir)"


