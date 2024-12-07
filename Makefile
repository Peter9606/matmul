IREE_HOME 	:= /home/peter/github/build.iluvatar.debug
COMPILE			:= $(IREE_HOME)/tools/iree-compile
RUN_MODULE 	:= $(IREE_HOME)/tools/iree-run-module
RUN_MLIR 		:= $(IREE_HOME)/tools/iree-run-mlir
TARGET 			:= iluvatar

input 			:= matmul
MLIR_SRC		:= $(input).mlir
LLIR_SRC 		:= $(input).llir

F16_DUMP_DIR:= /home/peter/github/playground/Matmul/dump-f16
I8_DUMP_DIR := /home/peter/github/playground/Matmul/dump-i8
COMMON_OPS 	:= --iree-hal-target-backends=iluvatar --iree-ilux-index-bits=32 --mlir-elide-resource-strings-if-larger=10
F16_OPS 		:= $(COMMON_OPS) --iree-hal-dump-executable-binaries-to=$(F16_DUMP_DIR) --iree-hal-dump-executable-intermediates-to=$(F16_DUMP_DIR)
I8_OPS 			:= $(COMMON_OPS) --iree-hal-dump-executable-binaries-to=$(I8_DUMP_DIR) --iree-hal-dump-executable-intermediates-to=$(I8_DUMP_DIR)


all: Makefile $(MLIR_SRC)
	$(COMPILE) --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 $(MLIR_SRC) -o $(LLIR_SRC) --mlir-print-ir-after-all --mlir-print-ir-before-all

cuda: Makefile $(MLIR_SRC)
	#$(COMPILE) --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir -o matmul.llir --mlir-print-ir-after-all
	#$(COMPILE) --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir -o matmul.llir --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all --mlir-print-ir-before-all
	#$(COMPILE) --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir --compile-to=vm --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul -o matmul.vmfb  --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all  2>&1 | tee cuda.print.after.all
	$(COMPILE) --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir --iree-hal-dump-executable-binaries-to=/home/peter/github/playground/Matmul/dump --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/dump -o dump/matmul.vmfb  --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee cuda.huge.print.after.all

iluvatar: Makefile $(MLIR_SRC)
	$(COMPILE) $(F16_OPS) matmul.mlir -o $(F16_DUMP_DIR)/matmul.vmfb

iluvatar-print: Makefile $(MLIR_SRC)
	#$(COMPILE) --compile-to=vm $(F16_OPS) -o matmul.vmfb  --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all  2>&1 | tee iluvatar.f16.print.after.all
	$(COMPILE) $(F16_OPS) matmul.mlir -o $(F16_DUMP_DIR)/matmul.vmfb --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee iluvatar.f16.print.after.all

iluvatar-torch: Makefile $(MLIR_SRC)
	$(COMPILE) $(F16_OPS) matmul.torch.mlir -o $(F16_DUMP_DIR)/matmul.vmfb 

iluvatar-torchprint: Makefile $(MLIR_SRC)
	#$(COMPILE) $(F16_OPS) --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 matmul.mlir --compile-to=vm --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/intermedia -o matmul.vmfb  --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all  2>&1 | tee iluvatar.f16.print.after.all
	$(COMPILE) $(F16_OPS) matmul.torch.mlir $(F16_OPS)  -o $(F16_DUMP_DIR)/matmul.vmfb --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee iluvatar.torch.f16.print.after.all



run-module: Makefile $(MLIR_SRC)
	#$(RUN_MODULE) --device=iluvatar --input="128x256xf16" --input="256x128xf16" --input="128x128xf32" --module=dump/matmul.vmfb
	#$(RUN_MODULE) --device=iluvatar --input="16x16xf16=1.0" --input="16x16xf16=1.0" --input="16x16xf32=1.0" --module=dump/matmul.vmfb
	$(RUN_MODULE) --device=iluvatar --input=@a.npy --input=@b.npy --input=@c.npy --module=dump-f16/matmul.vmfb --output=@res.npy

run-mlir: Makefile $(MLIR_SRC)
	#$(RUN_MLIR) --device=iluvatar --input="128x256xf16" --input="256x128xf16" --input="128x128xf32" matmul.mlir 
	$(RUN_MLIR) --device=iluvatar --input=@a.npy --input=@b.npy --input=@c.npy matmul.mlir

i8: Makefile $(MLIR_SRC)
	$(COMPILE) $(I8_OPS) matmul-i8.mlir -o $(I8_DUMP_DIR)/matmul-i8.vmfb


i8-print: Makefile $(MLIR_SRC)
	$(COMPILE) $(I8_OPS) matmul-i8.mlir -o $(I8_DUMP_DIR)/matmul.vmfb  --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee iluvatar.i8.print.after.all


clean:
	rm $(input).llir

profile-f16-4k8k8k:
	ixsys -o fgemm.4096x8192x8192.ptrace -t NVTX,CUDA,OSRT  python gen_data.py

profile-f16-4k4k4k:
	ixsys -o fgemm.4096x4096x4096.ptrace -t NVTX,CUDA,OSRT  python gen_data.py
