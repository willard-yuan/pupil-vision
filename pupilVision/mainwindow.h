#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QProgressBar>
#include <QTimer>
#include <QDebug>
#include "videohandler.h"
#include "calibrator.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static const int PROCESS_TIMEOUT = 50;

private:
    Ui::MainWindow *ui;
    QLabel* localStatus;
    QLabel* screenStatus;

    QTimer processTimer;
    Calibrator calibrator;

    VideoHandler videoHandler;
    void initForm();

    QMap<QString, QString> mapTrackingMethods;
    QString trackingMethod;

public slots:
    // general
    void exitClicked();
    void aboutClicked();

private slots:
    void process();
    void on_startButton_toggled(bool checked);
    void on_calibrateButton_toggled(bool checked);
    void on_recordButton_toggled(bool checked);
    void on_trackingComboBox_currentIndexChanged(const QString &arg1);
};

#endif // MAINWINDOW_H
