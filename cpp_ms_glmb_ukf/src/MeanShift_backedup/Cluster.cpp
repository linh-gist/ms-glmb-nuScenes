#include <cmath>
#include <vector>

#include "Cluster.hpp"
#include "Point.hpp"

Cluster::Cluster(Point centroid)
{
    this->centroid = std::move(centroid);
    checkPoints=std::vector<Point>(7);
    checkDis=std::vector<float>(7, std::numeric_limits<float>::max());
}

Point Cluster::getCentroid() const
{
    return centroid;
}

void Cluster::addPoint(Point point, int sensor, float distance)
{
    if(checkDis[sensor] > distance){
    points.emplace_back(point);
    checkPoints[sensor] = point;
        checkDis[sensor] = distance;
    }
}

long Cluster::getSize() const
{
    return points.size();
}

std::vector<Point>::iterator Cluster::begin()
{
    return points.begin();
}

std::vector<Point>::iterator Cluster::end()
{
    return points.end();
}

float Cluster::getSse() const
{
    float sum = 0.0;
    for (const Point &p : points)
        sum += std::pow(p.euclideanDistance(centroid), 2);
    return sum;
}
