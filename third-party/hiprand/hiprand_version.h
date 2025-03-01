// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPRAND_VERSION_H_
#define HIPRAND_VERSION_H_

/// \def HIPRAND_VERSION
/// \brief hipRAND library version
///
/// Version number may not be visible in the documentation.
///
/// HIPRAND_VERSION % 100 is the patch level,
/// HIPRAND_VERSION / 100 % 1000 is the minor version,
/// HIPRAND_VERSION / 100000 is the major version.
///
/// For example, if HIPRAND_VERSION is 100500, then
/// the major version is 1, the minor version is 5, and
/// the patch level is 0.
#define HIPRAND_VERSION 100500

#endif // HIPRAND_VERSION_H_
