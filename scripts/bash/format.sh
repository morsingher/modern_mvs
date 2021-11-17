#!/bin/bash

find -name '*.cc' -o -name '*.h' -o -name '*.cu' | xargs clang-format -i