#include "calibrator.h"

#include <QApplication>
#include <QDesktopWidget>
#include <QDebug>

Calibrator::Calibrator()
{
    state = None;
    padding = 50;
    buttonSize = 40;

    // get primary screen
    //screen = QApplication::desktop()->screenGeometry(1);
    screen = QApplication::desktop()->screenGeometry();

    // calibration points
    points.clear();
    int deltaNeg = padding-buttonSize/2;
    int deltaPos = padding+buttonSize/2;
    points.push_back(Point(deltaNeg, deltaNeg));
    points.push_back(Point(screen.width()-deltaPos, deltaNeg));
    points.push_back(Point(screen.width()-deltaPos, screen.height()-deltaPos));
    points.push_back(Point(deltaNeg, screen.height()-deltaPos));
}

void Calibrator::setMainWindow(MainWindow *p_mainWindow)
{
    mainWindow = p_mainWindow;
}

MainWindow* Calibrator::getMainWindow()
{
    return mainWindow;
}

void Calibrator::startCalibrating()
{
    qDebug("start calibrating");

    // create window
    window = new CalibrationWindow();
    window->setCalibrator(this);
    window->move(screen.center());

    // clear calibration points
    values.clear();

    // show next point
    showCalibrationPoint(0);

    // show fullscreen window
    window->showFullScreen();

    // enable calibrating
    state = Calibrating;
}

void Calibrator::dismissCalibration()
{
    qDebug("dismiss calibration");
    if (window != NULL)
    {
        window->close();
        delete window;
        window = NULL;
    }

    state = None;
}

Calibrator::CalibratorState Calibrator::getState()
{
    return state;
}

void Calibrator::setPosition(Point2f p_position)
{
    position = p_position;
}

Point Calibrator::calculatePosition(Point p_position, double* relativePercentX, double* relativePercentY)
{
    // save position
    setPosition(p_position);

    // values
    Point a = values[0];
    Point b = values[1];
    Point c = values[2];
    Point d = values[3];
    Point p = position;
    int screen_width = screen.width();
    int screen_height = screen.height();
    int calibPadding = padding;

    // calculate
    double C = (double)(a.y - p.y) * (d.x - p.x) - (double)(a.x - p.x) * (d.y - p.y);
    double B = (double)(a.y - p.y) * (c.x - d.x) + (double)(b.y - a.y) * (d.x - p.x) - (double)(a.x - p.x) * (c.y - d.y) - (double)(b.x - a.x) * (d.y - p.y);
    double A = (double)(b.y - a.y) * (c.x - d.x) - (double)(b.x - a.x) * (c.y - d.y);

    double D = B * B - 4 * A * C;

    double u = (-B - sqrt(D)) / (2 * A);

    double p1x = a.x + (b.x - a.x) * u;
    double p2x = d.x + (c.x - d.x) * u;

    double px = p.x;

    double v = (px - p1x) / (p2x - p1x);

    // in display bounds
    /*
    if (u > 1)
        u = 1;
    else if (u < 0)
        u = 0;

    if (v > 1)
        v = 1;
    else if (v < 0)
        v = 0;
    */

    Point ret;
    // calculate screen coordinates
    ret.x = u*(screen_width-2*calibPadding)+calibPadding;
    ret.y = v*(screen_height-2*calibPadding)+calibPadding;

    // save relative coordinates
    *relativePercentX = u;
    *relativePercentY = v;

    return ret;
}

Mat& Calibrator::drawCalibrationPoly(Mat &frame)
{
    if (frame.type() == CV_8U)
    {
        cvtColor(frame, frame, CV_GRAY2RGB);
    }

    // draw Eye region
    Scalar color(0,255,255);
    Point a = values[0];
    Point b = values[1];
    Point c = values[2];
    Point d = values[3];

    cv::line(frame, a, b, color, 2, 8);
    cv::line(frame, b, c, color, 2, 8);
    cv::line(frame, c, d, color, 2, 8);
    cv::line(frame, d, a, color, 2, 8);

    return frame;
}

void Calibrator::foundCalibrationPoint()
{
    // save current if not illeagl
    if (position.x != -1 && position.y != -1)
    {
        values.push_back(position);
    }

    // determine if finished
    if (values.size() < points.size())
    {
        // show next
        showCalibrationPoint(values.size());
    }
    else
    {
        qDebug("calibration finished");
        delete window;
        window = NULL;

        // set state
        state = Calibrated;
    }
}

void Calibrator::showCalibrationPoint(int p_index)
{
    window->setTargetPosition(points[p_index]);
}

QString Calibrator::getPositionString(Point local, Point pos)
{
    if (local.x == -1 && local.y == -1)
    {
        return QString("CLOSED");
    }
    else
    {
        return QString("(%1, %2)").arg(pos.x).arg(pos.y);
    }
}
