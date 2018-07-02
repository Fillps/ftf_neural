# C compiler
CC = g++ -std=c++11

destDir = /tmp/neural-sigmod

all: gen_bin gen_data mv_bin

gen_bin: neural-sigmod.cpp
	$(CC) neural-sigmod.cpp -o neural-sigmod

gen_data: gen_bin
	./neural-sigmod input.txt detect.log gold

mv_bin: gen_data
	mkdir -p $(destDir)
	mv neural-sigmod gold $(destDir)
	cp input.txt $(destDir)
	@echo "All Done, executables should be in the $(destDir)"


