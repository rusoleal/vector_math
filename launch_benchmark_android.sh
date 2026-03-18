#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Defaults (override via environment or flags)
# ---------------------------------------------------------------------------
ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-android-21}"
DEVICE_DIR="/data/local/tmp"
BINARY_NAME="vector_math_benchmark"

# ---------------------------------------------------------------------------
# Locate Android NDK
# ---------------------------------------------------------------------------
if [ -z "$NDK" ]; then
    SDK_ROOT="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
    NDK_BASE="$SDK_ROOT/ndk"
    if [ ! -d "$NDK_BASE" ]; then
        echo "error: NDK directory not found at $NDK_BASE"
        echo "  Set NDK=/path/to/ndk or ANDROID_HOME=/path/to/sdk"
        exit 1
    fi
    # Pick the newest installed NDK version
    NDK=$(ls -d "$NDK_BASE"/*/ 2>/dev/null | sort -V | tail -1)
    NDK="${NDK%/}"
fi

if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
    echo "error: could not locate Android NDK. Set NDK=/path/to/ndk"
    exit 1
fi

TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
    echo "error: toolchain file not found: $TOOLCHAIN"
    exit 1
fi

echo "NDK:      $NDK"
echo "ABI:      $ABI"
echo "Platform: $PLATFORM"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
BUILD_DIR="build/android-benchmark-$ABI"

cmake -B "$BUILD_DIR" -S . \
    -DVECTOR_MATH_BUILD_BENCHMARK=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$PLATFORM"

make -C "$BUILD_DIR"

# ---------------------------------------------------------------------------
# Verify adb and device
# ---------------------------------------------------------------------------
if ! command -v adb &>/dev/null; then
    echo "error: adb not found. Add Android SDK platform-tools to PATH."
    exit 1
fi

DEVICE_COUNT=$(adb devices | grep -c "device$" || true)
if [ "$DEVICE_COUNT" -eq 0 ]; then
    echo "error: no adb device/emulator detected. Start an emulator or connect a device."
    exit 1
fi

# ---------------------------------------------------------------------------
# Deploy and run
# ---------------------------------------------------------------------------
BINARY_PATH="$BUILD_DIR/$BINARY_NAME"

echo ""
echo "Pushing $BINARY_NAME to $DEVICE_DIR ..."
adb push "$BINARY_PATH" "$DEVICE_DIR/$BINARY_NAME"
adb shell chmod +x "$DEVICE_DIR/$BINARY_NAME"

echo "Running benchmark on device..."
echo ""
adb shell "$DEVICE_DIR/$BINARY_NAME"
