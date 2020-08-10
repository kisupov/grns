/*
 *  Logging functions
 */

#ifndef GRNS_TEST_LOGGER_CUH
#define GRNS_TEST_LOGGER_CUH

#include <iostream>

namespace Logger {

    // Tests headers
    enum TestHeader {
        TEST_VERIFY_EXTRANGE,
        TEST_VERIFY_MRC,
        TEST_VERIFY_RNSEVAL,
        TEST_PERF_RNSEVAL,
        TEST_VERIFY_RNSDIV,
        TEST_PERF_RNSDIV,
        TEST_PERF_CMP,
        TEST_VERIFY_MPINT,
        TEST_PERF_MPINT
    };

    const char *testHeaderAsString(enum TestHeader header) {
        switch (header) {
            case TEST_VERIFY_EXTRANGE:
                return "Test for checking the correctness of the extended-range floating-point routines";
            case TEST_VERIFY_MRC:
                return "Test for checking the correctness of the mixed-radix correction routines";
            case TEST_VERIFY_RNSEVAL:
                return "Test for checking the correctness and accuracy of the algorithms that calculate the RNS interval evaluation";
            case TEST_PERF_RNSEVAL:
                return "Test for measure the performance of the algorithms that calculate the RNS interval evaluation";
            case TEST_VERIFY_RNSDIV:
                return "Test for checking the RNS division algorithms";
            case TEST_PERF_RNSDIV:
                return "Test for measure the performance of the RNS division algorithms";
            case TEST_PERF_CMP:
                return "Test for measure the performance of the RNS magnitude comparison algorithms";
            case TEST_VERIFY_MPINT:
                return "Test for checking the correctness of the multiple-precision integer routines";
            case TEST_PERF_MPINT:
                return "Test for measure the performance of the multiple-precision integer routines";
        }
        return "";
    }

    // Служебные методы печати
    static void printSysdate() {
        time_t t = time(NULL);
        struct tm *tm = localtime(&t);
        std::cout << "Date: " << asctime(tm);
    }

    void printDash() {
        std::cout << "---------------------------------------------------" << std::endl;
    }

    static void printDDash() {
        std::cout << "===================================================" << std::endl;
    }

    static void printStars() {
        std::cout << "***************************************************" << std::endl;
    }

    static void printSpace() {
        std::cout << std::endl;
    }

    static void printString(const char *string) {
        std::cout << string << std::endl;
    }

    void printParam(const char *param, const char *value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const int value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const long value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void printParam(const char *param, const double value) {
        std::cout << "- " << param << ": " << value << std::endl;
    }

    void beginTestDescription(TestHeader header) {
        printSpace();
        printDDash();
        std::cout << testHeaderAsString(header) << std::endl;
        printSpace();
        printSysdate();
        printDDash();
    }

    void endTestDescription() {
        printDDash();
        printSpace();
        printSpace();
    }

    void printTestParameters(long operationSize, int numberOfRepeats) {
        printString("Parameters:");
        printParam("Operation size", operationSize);
        printParam("Number of repeats", numberOfRepeats);
        printDash();
    }

    void beginSection(const char *sectionName) {
        printString(sectionName);
    }

    void endSection(bool lastBeforeResults) {
        if (lastBeforeResults) {
            printDDash();
            printSpace();
        } else {
            printDash();
        }
    }
} // end of namespace Logger


#endif //GRNS_TEST_LOGGER_CUH