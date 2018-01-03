.PHONY: build, train

CUDA_VISIBLE_DEVICES=1
EXPERIMENT_DIR=../experiments
EXPERIMENT_BASE_NAME=test
EXPERIMENT_NAME=${EXPERIMENT_BASE_NAME}_$(shell date +'%Y.%m.%d-%H.%M.%S')
CHECKPOINT_ROOT=${EXPERIMENT_DIR}/checkpoints
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}
JSON_CONFIG=${EXPERIMENT_DIR}/config/default.json

build:
	mkdir -p cpp/build ; \
		cd cpp/build ; \
		CXX=g++-7 cmake .. ; \
		make ; \
		cp libembedding.so ../../core ; \
		cd ../.. ; \

train_parent:
	mkdir -p ${CHECKPOINT_DIR}; \
		CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 train.py \
		--random-seed 1234 \
		--json-config ${JSON_CONFIG} \
		--num-visual-images 1 \
		--save-pred-every 1000 \
		--detailed-summary-every 10 \
		--snapshot-dir ${CHECKPOINT_DIR} \
		2>&1 \
		| tee ${CHECKPOINT_DIR}/train.out -a

inference:
	CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python inference.py \
						 --random-seed 1234 \
						 --json-config ${JSON_CONFIG} \
						 --input-base-dir ${EXPERIMENT_DIR}/data/DAVIS2017 \
						 --sequence-name bear \
						 --num-visual-images 1 \
						 --detailed-summary-every 10 \
						 2>&1 \
						 | tee ${EXPERIMENT_DIR}/output/inference.out
