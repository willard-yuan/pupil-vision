#ifndef CALIBRATIONWINDOW_H
#define CALIBRATIONWINDOW_H

#include <QDialog>
#include <opencv2/opencv.hpp>

using namespace cv;

namespace Ui
{
class CalibrationWindow;
}

class Calibrator;

class CalibrationWindow : public QDialog
{
    Q_OBJECT
    
public:
    explicit CalibrationWindow(QWidget *parent = 0);
    ~CalibrationWindow();
    void reject();

    void setCalibrator(Calibrator*);
    void setTargetPosition(Point p);

public slots:
    void targetClicked();

private:
    Ui::CalibrationWindow *ui;
    Calibrator* calibrator;

};

#endif // CALIBRATIONWINDOW_H
