.PHONY: build

build:
	mkdir -p cpp/build ; \
		cd cpp/build ; \
		CXX=g++-7 cmake .. ; \
		make ; \
		cp libembedding.so ../../ ; \
		cd ../.. ; \
