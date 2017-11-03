#ifndef LIB_ERROR_H
#define LIB_ERROR_H

/* use gcc -Wall */
static void errTerminate(Boolean useExit3);
static void errPrint(Boolean use_errno, int errno, Boolean flush_stdout, va_list ap)

void errMessage(const char *format, ..,);
void errReport()

void errExit(const char *format) __attribute__((__noreturn__));
void errExitNoFlush(cosnt char *format) __attribute__((__noreturn__));

#endif
