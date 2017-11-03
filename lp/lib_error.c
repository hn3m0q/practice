#include <stdarg.h>
#include <errno.h>

#include "lib_error.h"

static void errReport(void)
{
    struct errno_rpt_struct
    {
        int val = ;
        char ename = ;
    }
    errno_rpt_struct errno_rpt
    return errno_rpt
}


static void errPrint(Boolean PRINT_ERRNO, int errno, const char *format, va_list ap)
{
    #define BUF_SIZE 500
    char buf[BUF_SIZE], errText[BUF_SIZE], userMsg[BUF_SIZE];

    vsnprintf(userMsg, BUF_SIZE, format, ap);

    if(PRINT_ERRNO) snprintf(errText, BUF_SIZE, "")
}




static void errPrintNoFlush()
{
    
}
