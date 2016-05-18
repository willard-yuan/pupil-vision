#include "recodewindow.h"
#include "ui_recodewindow.h"

#include "recorder.h"
#include "mainwindow.h"

recodewindow::recodewindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::recodewindow)
{
    ui->setupUi(this);
}

recodewindow::~recodewindow()
{
    delete ui;
}


void recodewindow::setRecorder(recorder* p_recorder)
{
    //recorder = p_recorder;
}
