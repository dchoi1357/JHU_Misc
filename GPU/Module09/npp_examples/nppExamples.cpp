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
int main(int argc, char *argv[]) {
	printf("%s Starting...\n\n", argv[0]);

	try {
		std::string sFilename;
		char *filePath;
		filePath = sdkFindFilePath("Lena.pgm", argv[0]);

		if (filePath) {
			sFilename = filePath;
		} else {
			sFilename = "Lena.pgm";
		}

		// if we specify the filename at the command line, then we only test sFilename[0].
		int file_errors = 0;
		std::ifstream infile(sFilename.data(), std::ifstream::in);

		if (infile.good()) {
			printf("%s opened: <%s> successfully!\n", argv[0], sFilename.data());
			infile.close();
		} else {
			printf("%s  unable to open: <%s> \n", argv[0], sFilename.data());
			file_errors++;
			infile.close();
			exit(EXIT_FAILURE);
		}

		std::string sResultFilename = sFilename;
		std::string::size_type dot = sResultFilename.rfind('.');

		if (dot != std::string::npos) {
			sResultFilename = sResultFilename.substr(0, dot);
		}
		sResultFilename += "_mirrored.pgm";

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
								oSizeROI, NPP_VERTICAL_AXIS) );

		// declare a host image for the result
		npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
		// and copy the device result data into it
		oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

		saveImage(sResultFilename, oHostDst);
		std::cout << "Saved image: " << sResultFilename << std::endl;

		nppiFree(oDeviceSrc.data());
		nppiFree(oDeviceDst.data());

		exit(EXIT_SUCCESS);

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

	return 0;
}
