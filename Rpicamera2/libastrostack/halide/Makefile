HALIDE_DIR := /home/admin/halide_build/Halide-21.0.0-arm-64-linux
TARGET      := arm-64-linux-arm_dot_prod-arm_fp16

CXX     := g++
CXXFLAGS := -std=c++17 -O2 -I$(HALIDE_DIR)/include
LDFLAGS  := -L$(HALIDE_DIR)/lib -lHalide -Wl,-rpath,$(HALIDE_DIR)/lib -lpthread -ldl

BUILD_DIR := build
GEN_DIR   := generated

.PHONY: all clean benchmark

all: $(GEN_DIR)/jsk_pipeline.a $(GEN_DIR)/denoise_med3x3.a $(GEN_DIR)/guided_filter.a jsk_halide.so

# ─── Étape 1 : compiler les generator executables ─────────────────────────────
$(BUILD_DIR)/jsk_gen: jsk_pipeline.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(HALIDE_DIR)/lib/libHalide_GenGen.a \
	    $(LDFLAGS) -o $@

$(BUILD_DIR)/denoise_gen: denoise_pipeline.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(HALIDE_DIR)/lib/libHalide_GenGen.a \
	    $(LDFLAGS) -o $@

$(BUILD_DIR)/guided_gen: guided_filter_pipeline.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(HALIDE_DIR)/lib/libHalide_GenGen.a \
	    $(LDFLAGS) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(GEN_DIR):
	mkdir -p $(GEN_DIR)

# ─── Étape 2 : générer le code AOT (4 variantes pour les combinaisons HDR) ────
$(GEN_DIR)/jsk_pipeline.a: $(BUILD_DIR)/jsk_gen | $(GEN_DIR)
	@echo "==> Génération AOT median-3 niveaux"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/jsk_gen \
	    -g jsk_pipeline \
	    -o $(GEN_DIR) \
	    -f jsk_pipeline_med3 \
	    target=$(TARGET) \
	    hdr_levels=3 hdr_method=1
	@echo "==> Génération AOT mean-3 niveaux"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/jsk_gen \
	    -g jsk_pipeline \
	    -o $(GEN_DIR) \
	    -f jsk_pipeline_mean3 \
	    target=$(TARGET) \
	    hdr_levels=3 hdr_method=0
	@echo "==> Génération AOT median-2 niveaux"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/jsk_gen \
	    -g jsk_pipeline \
	    -o $(GEN_DIR) \
	    -f jsk_pipeline_med2 \
	    target=$(TARGET) \
	    hdr_levels=2 hdr_method=1
	@echo "==> Génération AOT bypass (1 niveau = pas de HDR)"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/jsk_gen \
	    -g jsk_pipeline \
	    -o $(GEN_DIR) \
	    -f jsk_pipeline_bypass \
	    target=$(TARGET) \
	    hdr_levels=2 hdr_method=0
	touch $(GEN_DIR)/jsk_pipeline.a
	@echo "==> Fichiers générés :"
	ls -lh $(GEN_DIR)/

# ─── Étape 2b : générer le code AOT médiane 3×3 ──────────────────────────────
$(GEN_DIR)/denoise_med3x3.a: $(BUILD_DIR)/denoise_gen | $(GEN_DIR)
	@echo "==> Génération AOT médiane 3×3 (canal unique uint8)"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/denoise_gen \
	    -g denoise_pipeline \
	    -o $(GEN_DIR) \
	    -f denoise_med3x3 \
	    target=$(TARGET)
	@echo "==> denoise_med3x3.a généré"

# ─── Étape 2c : générer le code AOT guided filter ────────────────────────────
$(GEN_DIR)/guided_filter.a: $(BUILD_DIR)/guided_gen | $(GEN_DIR)
	@echo "==> Génération AOT guided filter (radius=8)"
	LD_LIBRARY_PATH=$(HALIDE_DIR)/lib $(BUILD_DIR)/guided_gen \
	    -g guided_filter_pipeline \
	    -o $(GEN_DIR) \
	    -f guided_filter \
	    target=$(TARGET) \
	    radius=8
	@echo "==> guided_filter.a généré"

# ─── Étape 3 : créer le .so Python-callable ──────────────────────────────────
jsk_halide.so: $(GEN_DIR)/jsk_pipeline.a $(GEN_DIR)/denoise_med3x3.a $(GEN_DIR)/guided_filter.a jsk_wrapper.cpp
	$(CXX) -std=c++17 -O2 -shared -fPIC \
	    -I$(HALIDE_DIR)/include \
	    -I$(GEN_DIR) \
	    jsk_wrapper.cpp \
	    $(GEN_DIR)/jsk_pipeline_med3.a \
	    $(GEN_DIR)/jsk_pipeline_mean3.a \
	    $(GEN_DIR)/jsk_pipeline_med2.a \
	    $(GEN_DIR)/jsk_pipeline_bypass.a \
	    $(GEN_DIR)/denoise_med3x3.a \
	    $(GEN_DIR)/guided_filter.a \
	    $(HALIDE_DIR)/lib/libHalide.a \
	    -lpthread -ldl \
	    -o jsk_halide.so
	@echo "==> jsk_halide.so créé"

# ─── Benchmark standalone ────────────────────────────────────────────────────
$(BUILD_DIR)/benchmark: benchmark.cpp $(GEN_DIR)/jsk_pipeline.a
	$(CXX) -std=c++17 -O2 \
	    -I$(HALIDE_DIR)/include \
	    -I$(GEN_DIR) \
	    benchmark.cpp \
	    $(GEN_DIR)/jsk_pipeline_med3.a \
	    $(GEN_DIR)/jsk_pipeline_mean3.a \
	    $(HALIDE_DIR)/lib/libHalide.a \
	    -lpthread -ldl \
	    -o $(BUILD_DIR)/benchmark

benchmark: $(BUILD_DIR)/benchmark
	$(BUILD_DIR)/benchmark

clean:
	rm -rf $(BUILD_DIR) $(GEN_DIR) jsk_halide.so
