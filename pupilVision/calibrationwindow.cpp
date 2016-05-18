#include "calibrationwindow.h"
#include "ui_calibrationwindow.h"
#include "calibrator.h"

#include <QDebug>
#include "mainwindow.h"

CalibrationWindow::CalibrationWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CalibrationWindow)
{
    ui->setupUi(this);
}

CalibrationWindow::~CalibrationWindow()
{
    delete ui;
}

void CalibrationWindow::setCalibrator(Calibrator* p_calibrator)
{
    calibrator = p_calibrator;
}

void CalibrationWindow::targetClicked()
{
    calibrator->foundCalibrationPoint();
}

void CalibrationWindow::setTargetPosition(Point p)
{
    ui->target->move(p.x, p.y);
}

// do not close
void CalibrationWindow::reject()
{
    qDebug("reject calibration");
    //calibrator->getMainWindow()->toggleCalibrate(false);
}
