#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "helper.h"

#include "pupilTrack/PupilTracker.h"
#include "pupilTrack/cvx.h"
#include "pupilTrack/erase_specular.h"
#include "pupilTrack/utils.h"

#include "pupilLib/pupilDetect.hpp"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->initForm();

    calibrator.setMainWindow(this);

    // 设置皮肤
    QString qss;
    QFile qssFile(":/qss/black.css");
    qssFile.open(QFile::ReadOnly);
    if(qssFile.isOpen()){
        qss = QLatin1String(qssFile.readAll());
        qApp->setStyleSheet(qss);
        qssFile.close();
    }

    mapTrackingMethods["重心法"] = QString("gravityMethod");
    mapTrackingMethods["椭圆法"] = QString("ellipseMethod");

    this->trackingMethod = "gravityMethod";  //设置默认的跟踪方法

    // local status widget
    localStatus = new QLabel(this);
    localStatus->setHidden(true);
    ui->statusBar->addPermanentWidget(localStatus);

    // screen status widget
    screenStatus = new QLabel(this);
    screenStatus->setHidden(true);
    ui->statusBar->addPermanentWidget(screenStatus);

    // timer
    QObject::connect(&processTimer, SIGNAL(timeout()), this, SLOT(process()));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::initForm()
{
    QStringList trackingMethods,calibrationMethods;
    trackingMethods << "重心法" << "椭圆法";
    calibrationMethods << "线性模型" << "二元齐次模型";
    ui->trackingComboBox->addItems(trackingMethods);
    ui->calibrationComboBox->addItems(calibrationMethods);
}


// render loop
void MainWindow::process()
{
    Mat frame = videoHandler.getFrame();
    Mat image = frame;

    cv::Point2f localPos;

    if(this->trackingMethod == "ellipseMethod"){
        pupiltracker::findPupilEllipse_out out;
        pupiltracker::TrackerParams params;
        params.Radius_Min = 30;
        params.Radius_Max = 80;
        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.PercentageInliers = 30;
        //params.InlierIterations = 2;
        params.InlierIterations = 1;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;
        bool eraseSpecular = true;
        pupiltracker::findPupilEllipse(params, image, out, eraseSpecular);
        pupiltracker::cvx::cross(frame, out.pPupil, 1, pupiltracker::cvx::rgb(255, 255, 0)); // 画椭圆中心
        cv::ellipse(frame, out.elPupil, pupiltracker::cvx::rgb(255, 0, 0), 1); // 画椭圆
        localPos = out.pPupil;
    }else{
        cv::Mat_<uchar> mEye;
        cv::cvtColor(frame, mEye, cv::COLOR_BGR2GRAY);
        erase_specular(mEye);
        localPos = pupilDetect(mEye);
        cv::circle(frame, localPos, 2, cv::Scalar(0, 0, 255), 4);
    }

    // set video
    ui->videoCanvas->setPixmap(QPixmap::fromImage(Helper::mat2qimage(frame)).scaled(640, 480));
    // draw status
    localStatus->setHidden(false);
    localStatus->setText("眼睛坐标:"+calibrator.getPositionString(localPos, localPos));

    // screen positions
    Point2f screenPos;
    double relativeX;
    double relativeY;

    // ------------------
    // CALIBRATION STATES
    switch (calibrator.getState())
    {
    // while calibrating
    case Calibrator::Calibrating:

        // add position info
        calibrator.setPosition(localPos);
        break;

    case Calibrator::Calibrated:

        // render quadriliteral
        frame = calibrator.drawCalibrationPoly(frame);

        // calculate position
        screenPos = calibrator.calculatePosition(localPos, &relativeX, &relativeY);

        // status
        screenStatus->setHidden(false);
        screenStatus->setText("屏幕坐标:"+calibrator.getPositionString(localPos, screenPos));

        break;

    default:
        screenStatus->setHidden(true);
        break;
    }
}

// ----------------------------------------------------------------------------
// general slots
void MainWindow::exitClicked()
{
    qDebug("exit");
    QApplication::quit();
}

void MainWindow::aboutClicked()
{
    qDebug("about");
}

// ----------------------------------------------------------------------------
void MainWindow::on_startButton_toggled(bool checked)
{
    if (checked)
    {
        // switching the video ON
        qDebug("video: ON");
        videoHandler.start(0);
        processTimer.start(PROCESS_TIMEOUT);
        ui->videoCanvas->setHidden(false);
        ui->calibrateButton->setEnabled(true);
        ui->recordButton->setEnabled(true);
    }
    else
    {
        if (ui->calibrateButton->isChecked())
        {
            ui->calibrateButton->click();
        }
        ui->calibrateButton->setEnabled(false);
        ui->recordButton->setEnabled(false);

        // switching the video OFF
        qDebug("video: OFF");
        processTimer.stop();
        videoHandler.stop();
        ui->videoCanvas->setHidden(false);
        localStatus->setHidden(true);
    }
}


// 校正
void MainWindow::on_calibrateButton_toggled(bool checked)
{
    if (checked){
        ui->recordButton->setEnabled(true);
        qDebug("calibrate: ON");
        calibrator.startCalibrating();
    }
    else{
        if (ui->recordButton->isChecked())
        {
            ui->recordButton->click();
        }
        ui->recordButton->setEnabled(false);
        qDebug("calibrate: OFF");
        // disable calibrating
        calibrator.dismissCalibration();
    }
    return;
}

void MainWindow::on_recordButton_toggled(bool checked)
{
    // re-set state
    //ui->recordButton->setChecked(state);

    if (checked)
    {
        qDebug("record: ON");

        if (QApplication::desktop()->screenCount() == 1)
        {
            //showMinimized();
            //QTimer::singleShot(1000, this, SLOT(startRecording()));
        }
        else
        {
            // start immediately
            //startRecording();
        }
    }
    else
    {
        qDebug("record: OFF");

        if (QApplication::desktop()->screenCount() == 1)
        {
            setWindowState( (windowState() & ~Qt::WindowMinimized) | Qt::WindowActive);
            //activateWindow();

            //QTimer::singleShot(500, this, SLOT(stopRecording()));
        }
        else
        {
            //stopRecording();
        }
    }

    return;
}

//选取跟踪方法
void MainWindow::on_trackingComboBox_currentIndexChanged(const QString &arg1)
{
    QString trackingChoosedMethod = arg1.trimmed();

    this->trackingMethod = mapTrackingMethods[trackingChoosedMethod];

    if (this->trackingMethod.trimmed() == "")
        this->trackingMethod = "gravityMethod";
}
