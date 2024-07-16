IREE_HOME 	:= /home/peter/github/iree-debug
COMPILE			:= $(IREE_HOME)/tools/iree-compile
TARGET 			:= iluvatar

input 			:= matmul
MLIR_SRC		:= $(input).mlir
LLIR_SRC 		:= $(input).llir

all: Makefile $(MLIR_SRC)
	$(COMPILE) --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 $(MLIR_SRC) -o $(LLIR_SRC) --mlir-print-ir-after-all --mlir-print-ir-before-all

cuda: Makefile $(MLIR_SRC)
	#/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir -o matmul.llir --mlir-print-ir-after-all
	#/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir -o matmul.llir --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all --mlir-print-ir-before-all
	#/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir --compile-to=vm --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul -o matmul.vmfb  --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all  2>&1 | tee cuda.print.after.all
	/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 --mlir-elide-resource-strings-if-larger=10 matmul.mlir --iree-hal-dump-executable-binaries-to=/home/peter/github/playground/Matmul/dump --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/dump -o dump/matmul.vmfb  --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee cuda.huge.print.after.all

iluvatar: Makefile $(MLIR_SRC)
	/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 matmul.mlir --iree-hal-dump-executable-binaries-to=/home/peter/github/playground/Matmul/dump --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/dump -o dump/matmul.vmfb


iluvatar-print: Makefile $(MLIR_SRC)
	#/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 matmul.mlir --compile-to=vm --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/intermedia -o matmul.vmfb  --mlir-print-ir-module-scope --mlir-disable-threading --mlir-print-ir-after-all  2>&1 | tee iluvatar.print.after.all
	/home/peter/github/build.iluvatar.debug/tools/iree-compile --iree-hal-target-backends=iluvatar --mlir-elide-resource-strings-if-larger=10 matmul.mlir --iree-hal-dump-executable-binaries-to=/home/peter/github/playground/Matmul/dump --iree-hal-dump-executable-intermediates-to=/home/peter/github/playground/Matmul/dump -o dump/matmul.vmfb  --mlir-print-ir-module-scope --mlir-print-ir-after-all  2>&1 | tee iluvatar.print.after.all

run-module: Makefile $(MLIR_SRC)
	#/home/peter/github/build.iluvatar.debug/tools/iree-run-module --device=iluvatar --input="128x256xf16" --input="256x128xf16" --input="128x128xf32" --module=dump/matmul.vmfb
	/home/peter/github/build.iluvatar.debug/tools/iree-run-module --device=iluvatar --input="16x16xf16=1.0" --input="16x16xf16=1.0" --input="16x16xf32=1.0" --module=dump/matmul.vmfb

run-mlir: Makefile $(MLIR_SRC)
	/home/peter/github/build.iluvatar.debug/tools/iree-run-mlir --device=iluvatar --input="128x256xf16" --input="256x128xf16" --input="128x128xf32" matmul.mlir 


clean:
	rm $(input).llir

