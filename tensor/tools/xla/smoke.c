// Loads a PJRT plugin with dlopen and checks that it answers: the API version
// matches the header, and a client is created and destroyed.
//   cc -I <dir of pjrt_c_api.h> smoke.c -ldl -o smoke && ./smoke <plugin.so>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "pjrt_c_api.h"

static int failed(const PJRT_Api *api, PJRT_Error *err, const char *what) {
  if (err == NULL) return 0;
  PJRT_Error_Message_Args msg;
  memset(&msg, 0, sizeof msg);
  msg.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg.error = err;
  api->PJRT_Error_Message(&msg);
  fprintf(stderr, "smoke: %s: %.*s\n", what, (int)msg.message_size, msg.message);
  PJRT_Error_Destroy_Args destroy;
  memset(&destroy, 0, sizeof destroy);
  destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = err;
  api->PJRT_Error_Destroy(&destroy);
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: smoke <pjrt plugin .so>\n");
    return 2;
  }
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) {
    fprintf(stderr, "smoke: dlopen: %s\n", dlerror());
    return 1;
  }
  const PJRT_Api *(*get_api)(void) = (const PJRT_Api *(*)(void))dlsym(lib, "GetPjrtApi");
  if (get_api == NULL) {
    fprintf(stderr, "smoke: dlsym GetPjrtApi: %s\n", dlerror());
    return 1;
  }
  const PJRT_Api *api = get_api();
  if (api == NULL) {
    fprintf(stderr, "smoke: GetPjrtApi returned NULL\n");
    return 1;
  }
  int major = api->pjrt_api_version.major_version;
  int minor = api->pjrt_api_version.minor_version;
  printf("PJRT API %d.%d\n", major, minor);
  if (major != PJRT_API_MAJOR || minor != PJRT_API_MINOR) {
    fprintf(stderr, "smoke: the header says %d.%d\n", PJRT_API_MAJOR, PJRT_API_MINOR);
    return 1;
  }

  PJRT_Plugin_Initialize_Args init;
  memset(&init, 0, sizeof init);
  init.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  if (failed(api, api->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize")) return 1;

  PJRT_Client_Create_Args create;
  memset(&create, 0, sizeof create);
  create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (failed(api, api->PJRT_Client_Create(&create), "PJRT_Client_Create")) return 1;
  if (create.client == NULL) {
    fprintf(stderr, "smoke: PJRT_Client_Create returned no client\n");
    return 1;
  }

  PJRT_Client_PlatformName_Args name;
  memset(&name, 0, sizeof name);
  name.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  name.client = create.client;
  if (failed(api, api->PJRT_Client_PlatformName(&name), "PJRT_Client_PlatformName")) return 1;
  printf("platform %.*s\n", (int)name.platform_name_size, name.platform_name);

  PJRT_Client_Destroy_Args destroy;
  memset(&destroy, 0, sizeof destroy);
  destroy.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy.client = create.client;
  if (failed(api, api->PJRT_Client_Destroy(&destroy), "PJRT_Client_Destroy")) return 1;
  printf("ok\n");
  return 0;
}
