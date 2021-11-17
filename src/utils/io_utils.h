#ifndef MVS_IO_UTILS_H
#define MVS_IO_UTILS_H

#include "common.h"

std::vector<FramePtr> readFrames(const std::string& path);
bool checkInputData(const std::string& folder);   // TODO: move to options?
void createOutputDir(const std::string& folder);  // TODO: move to options?
std::string getFilename(const std::string& folder, const int id, const std::string& ext);

#endif