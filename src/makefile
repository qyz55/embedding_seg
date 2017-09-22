.PHONY: build, train

build:
	mkdir -p cpp/build ; \
		cd cpp/build ; \
		CXX=g++-7 cmake .. ; \
		make ; \
		cp libembedding.so ../../ ; \
		cd ../.. ; \

train:
	CUDA_VISIBLE_DEVICES=0 python train.py
