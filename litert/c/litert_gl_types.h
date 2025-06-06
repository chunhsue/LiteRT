// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_C_LITERT_GL_TYPES_H_
#define ODML_LITERT_LITERT_C_LITERT_GL_TYPES_H_

#include <stdint.h>
#if LITERT_HAS_OPENGL_SUPPORT
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES3/gl32.h>
#endif  // LITERT_HAS_OPENGL_SUPPORT

#ifdef __cplusplus
extern "C" {
#endif

#if LITERT_HAS_OPENGL_SUPPORT
typedef GLenum LiteRtGLenum;
typedef GLuint LiteRtGLuint;
typedef GLint LiteRtGLint;
#else
// Allows for compilation of GL types when OpenGl support is not available.
typedef uint32_t LiteRtGLenum;
typedef uint32_t LiteRtGLuint;
typedef int32_t LiteRtGLint;
typedef void *EGLContext;
typedef void *EGLDisplay;
typedef void *EGLSyncKHR;
#define EGL_NO_CONTEXT static_cast<EGLContext>(0)
#define EGL_NO_DISPLAY static_cast<EGLDisplay>(0)
#endif  // LITERT_HAS_OPENGL_SUPPORT

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ODML_LITERT_LITERT_C_LITERT_GL_TYPES_H_
