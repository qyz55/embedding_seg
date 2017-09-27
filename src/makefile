.PHONY: build, train

CUDA_VISIBLE_DEVICES=1
EXPERIMENT_DIR=../experiments
EXPERIMENT_BASE_NAME=embedding
EXPERIMENT_NAME=${EXPERIMENT_BASE_NAME}_$(shell date +'%Y.%m.%d-%H.%M.%S')
CHECKPOINT_ROOT=${EXPERIMENT_DIR}/checkpoints
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}

build:
	mkdir -p cpp/build ; \
		cd cpp/build ; \
		CXX=g++-7 cmake .. ; \
		make ; \
		cp libembedding.so ../../core ; \
		cd ../.. ; \

train:
	mkdir -p ${CHECKPOINT_DIR}; \
		CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train.py \
		--random-seed 1234 \
		--json-config ../experiments/config/default.json \
		--num-visual-images 2 \
		--save-pred-every 1000 \
		--detailed-summary-every 500 \
		--snapshot-dir ${CHECKPOINT_DIR} \
		| tee ${CHECKPOINT_DIR}/train.out -a
