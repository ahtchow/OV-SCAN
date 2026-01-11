#!/bin/bash
set -e  # Exit on error

# ==============================================================================
# OV-SCAN Setup Script
# ==============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==============================================================================
# Function: Setup DCNv4
# ==============================================================================
setup_dcnv4() {
    echo ""
    echo "========================================="
    echo "Setting up DCNv4 (Deformable Conv v4)..."
    echo "========================================="
    
    if [ ! -d "/OV-SCAN/repos/DCNv4" ]; then
        echo -e "${RED}✗ DCNv4 repository not found at /OV-SCAN/repos/DCNv4${NC}"
        return 1
    fi
    
    cd /OV-SCAN/repos/DCNv4/DCNv4_op/
    python3 setup.py build_ext --inplace
    pip install -e . --no-user
    cd /OV-SCAN
    
    echo -e "${GREEN}✓ DCNv4 setup complete${NC}"
}

# ==============================================================================
# Function: Setup NuScenes Devkit
# ==============================================================================
setup_nuscenes() {
    echo ""
    echo "========================================="
    echo "Setting up NuScenes Devkit..."
    echo "========================================="
    
    if [ ! -d "/OV-SCAN/repos/nuscenes-devkit" ]; then
        echo -e "${RED}✗ nuscenes-devkit repository not found${NC}"
        return 1
    fi
    
    export PYTHONPATH=$PYTHONPATH:/OV-SCAN/repos/nuscenes-devkit/python-sdk
    echo -e "${GREEN}✓ NuScenes Devkit setup complete${NC}"
}

# ==============================================================================
# Function: Setup ICP-Flow and Patchwork++
# ==============================================================================
setup_icp_flow() {
    echo ""
    echo "========================================="
    echo "Setting up ICP-Flow & Patchwork++..."
    echo "========================================="
    
    if [ ! -d "/OV-SCAN/repos/ICP-Flow" ]; then
        echo -e "${RED}✗ ICP-Flow repository not found${NC}"
        return 1
    fi
    
    # Build patchwork-plusplus Python wrapper
    cd /OV-SCAN/repos/ICP-Flow/patchwork-plusplus
    rm -rf build
    mkdir -p build
    cd build
    
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    
    cd /OV-SCAN
    
    # Add to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:/OV-SCAN/repos/ICP-Flow
    export PYTHONPATH=$PYTHONPATH:/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/python_wrapper
    
    echo -e "${GREEN}✓ ICP-Flow setup complete${NC}"
}

# ==============================================================================
# Function: Setup ImmortalTracker
# ==============================================================================
setup_immortaltracker() {
    echo ""
    echo "========================================="
    echo "Setting up ImmortalTracker..."
    echo "========================================="
    
    if [ ! -d "/OV-SCAN/repos/ImmortalTracker" ]; then
        echo -e "${RED}✗ ImmortalTracker repository not found${NC}"
        return 1
    fi
    
    export PYTHONPATH=$PYTHONPATH:/OV-SCAN/repos/ImmortalTracker
    echo -e "${GREEN}✓ ImmortalTracker setup complete${NC}"
}

# ==============================================================================
# Function: Verify Installations
# ==============================================================================
verify_installations() {
    echo ""
    echo "========================================="
    echo "Verifying Installation..."
    echo "========================================="
    
    local all_success=true
    
    # Test pypatchworkpp
    if python3 -c "import pypatchworkpp" 2>/dev/null; then
        echo -e "${GREEN}✓ pypatchworkpp imported successfully${NC}"
    else
        echo -e "${RED}✗ pypatchworkpp import failed${NC}"
        all_success=false
    fi
    
    # Test DCNv4
    if python3 -c "import DCNv4" 2>/dev/null; then
        echo -e "${GREEN}✓ DCNv4 imported successfully${NC}"
    else
        echo -e "${RED}✗ DCNv4 import failed${NC}"
        all_success=false
    fi
    
    # Test nuscenes-devkit
    if python3 -c "from nuscenes.nuscenes import NuScenes" 2>/dev/null; then
        echo -e "${GREEN}✓ nuscenes-devkit imported successfully${NC}"
    else
        echo -e "${RED}✗ nuscenes-devkit import failed${NC}"
        all_success=false
    fi
    
    # Test ImmortalTracker
    if python3 -c "import sys; sys.path.insert(0, '/OV-SCAN/repos/ImmortalTracker'); import mot_3d" 2>/dev/null; then
        echo -e "${GREEN}✓ ImmortalTracker imported successfully${NC}"
    else
        echo -e "${RED}✗ ImmortalTracker import failed${NC}"
        all_success=false
    fi
    
    if [ "$all_success" = false ]; then
        echo -e "${YELLOW}⚠ Some components failed verification${NC}"
        return 1
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================
main() {
    echo ""
    echo "========================================="
    echo "OV-SCAN Environment Setup"
    echo "========================================="
    
    # Run setup functions
    setup_dcnv4
    setup_nuscenes
    setup_icp_flow
    setup_immortaltracker
    
    # Verify everything
    verify_installations
    
    echo ""
    echo "========================================="
    echo -e "${GREEN}✓ OV-SCAN Environment Ready!${NC}"
    echo "========================================="
}

# Run main function
main
