#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

void mirrorFunc(const char* filePath, NppiAxis mirrorType, std::string suff) {
	std::string sFilename = filePath;

	// if we specify the filename at the command line, then we only test sFilename[0].
	int file_errors = 0;
	std::ifstream infile(sFilename.data(), std::ifstream::in);

	if (infile.good()) {
		printf("\topened: <%s> successfully!\n", sFilename.data());
		infile.close();
	} else {
		printf("\tunable to open: <%s> \n", sFilename.data());
		file_errors++;
		infile.close();
		exit(EXIT_FAILURE);
	}

	std::string sResultFilename = sFilename;
	std::string::size_type dot = sResultFilename.rfind('.');

	if (dot != std::string::npos) {
		sResultFilename = sResultFilename.substr(0, dot);
	}
	sResultFilename += ("_mirrored_" + suff + ".pgm");

	// declare a host image object for an 8-bit grayscale image
	npp::ImageCPU_8u_C1 oHostSrc;
	// load gray-scale image from disk
	npp::loadImage(sFilename, oHostSrc);
	// declare a device image and copy construct from the host image,
	// i.e. upload host to device
	npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

	// create struct with ROI size
	NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
	
	// allocate device image of appropriately reduced size
	npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
	
	NPP_CHECK_NPP ( // run mirroring
		nppiMirror_8u_C1R(	oDeviceSrc.data(), oDeviceSrc.pitch(),
							oDeviceDst.data(), oDeviceDst.pitch(),
							oSizeROI, mirrorType) );

	// declare a host image for the result
	npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
	// and copy the device result data into it
	oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

	saveImage(sResultFilename, oHostDst);
	std::cout << "\tSaved image: " << sResultFilename << std::endl;

	nppiFree(oDeviceSrc.data());
	nppiFree(oDeviceDst.data());
}


int main(int argc, char *argv[]) {
	char *filePath;
	int flipType;
	if (argc == 2) { // if an input file was specified
		filePath = sdkFindFilePath(argv[1], argv[0]);
		flipType = 1;
	} else if (argc > 2) {
		filePath = sdkFindFilePath(argv[1], argv[0]);
		flipType = std::max( std::min(atoi(argv[2]), 3), 1);
	} else {
		filePath = sdkFindFilePath("Lena.pgm", argv[0]); 
		flipType = 1;
	}
	
	const char *flipName;
	std::string fSuffix;
	NppiAxis ax;
	if (flipType == 1) {
		flipName = "horizontally";
		fSuffix = "horiz";
		ax = NPP_HORIZONTAL_AXIS;
	} else if (flipType == 2) {
		flipName = "vertically";
		fSuffix = "vert";
		ax = NPP_VERTICAL_AXIS;
	} else {
		flipName = "horizontally and vertically";
		fSuffix = "horiz_vert"; 
		ax = NPP_BOTH_AXIS;
	}
	
	
	try {
		printf("Mirroring %s %s...\n",  filePath, flipName);
		mirrorFunc(filePath, ax, fSuffix);
	} catch (npp::Exception &rException){
		std::cerr << "Program error! The following exception occurred: \n";
		std::cerr << rException << std::endl;
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
	} catch (...) {
		std::cerr << "Program error! An unknow type of exception occurred. \n";
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
		return -1;
	}
	exit(EXIT_SUCCESS);
	return 0;
}
