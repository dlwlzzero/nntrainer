#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef REMOTE_HANDLE64_TYPEDEF
#define REMOTE_HANDLE64_TYPEDEF
typedef uint64_t remote_handle64;
#endif

int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();
remote_handle64 get_global_handle();
void init_htp_backend();
int create_htp_message_channel(int fd, unsigned int max_msg_size);
int alloc_shared_mem_buf(void **p_buf, int *p_fd, size_t size);
void free_shared_mem_buf(void *buf, int fd, size_t size);

#ifdef __cplusplus
}
#endif
