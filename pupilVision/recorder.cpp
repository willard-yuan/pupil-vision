#include "recorder.h"

#include <QDebug>
#include <QApplication>
#include <QDesktopWidget>

recorder::recorder()
{

}

void recorder::setMainWindow(MainWindow *p_mainWindow)
{
    mainWindow = p_mainWindow;
}


void recorder::startRecording()
{
    QRect screen = QApplication::desktop()->screenGeometry(1);

    window = new recodewindow();
    window->setRecorder(this);
    window->move(screen.center());
    window->showFullScreen();
}

