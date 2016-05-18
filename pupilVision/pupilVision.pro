#-------------------------------------------------
#
# Project created by QtCreator 2016-05-15T08:52:04
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = pupilVision
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    videohandler.cpp \
    helper.cpp \
    pupilLib/findEyeCenter.cpp \
    pupilLib/helpers.cpp \
    pupilLib/pupilDetect.cpp \
    pupilTrack/cvx.cpp \
    pupilTrack/utils.cpp \
    pupilTrack/PupilTracker.cpp \
    pupilTrack/erase_specular.cpp \
    calibrationwindow.cpp \
    calibrator.cpp

HEADERS  += mainwindow.h \
    videohandler.h \
    helper.h \
    pupilLib/constants.h \
    pupilLib/erase_specular.h \
    pupilLib/findEyeCenter.h \
    pupilLib/helpers.h \
    pupilLib/pupilDetect.hpp \
    pupilTrack/ConicSection.h \
    pupilTrack/cvx.h \
    pupilTrack/EllipseRansac.h \
    pupilTrack/utils.h \
    pupilTrack/PupilTracker.h \
    pupilTrack/erase_specular.h \
    calibrator.h \
    calibrationwindow.h

FORMS    += \
    calibrationwindow.ui \
    recodewindow.ui \
    mainwindow.ui

QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.10

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2
INCLUDEPATH += /usr/local/include

LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_photo #-ltbb -ltbbmalloc -ltbbmalloc_proxy

RESOURCES += design.qrc
