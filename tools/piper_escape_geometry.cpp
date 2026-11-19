// Read-only full-mesh escape metric. No publishers, controllers or scene writes.
#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/collision_detection_fcl/collision_common.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <fcl/fcl.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>

using Geometry = collision_detection::FCLGeometryConstPtr;
int main(int argc, char** argv) {
  ros::init(argc, argv, "piper_escape_geometry", ros::init_options::AnonymousName);
  try {
    robot_model_loader::RobotModelLoader loader("robot_description");
    auto model = loader.getModel();
    if (!model) throw std::runtime_error("robot model missing");
    auto arm = model->getLinkModel("piper_link2");
    auto panel = model->getLinkModel("top_chassis_link");
    if (!arm || !panel) throw std::runtime_error("escape links missing");
    ros::NodeHandle nh;
    std::string original_description;
    if(!nh.getParam("robot_description", original_description)) throw std::runtime_error("URDF unavailable");
    auto client = nh.serviceClient<moveit_msgs::GetPlanningScene>("/get_planning_scene");
    if (!client.waitForExistence(ros::Duration(5))) throw std::runtime_error("scene unavailable");
    std::vector<Geometry> arm_geom, panel_geom;
    double saved_padding = -1, saved_scale = -1;
    for (size_t i=0;i<arm->getShapes().size();++i)
      arm_geom.push_back(collision_detection::createCollisionGeometry(arm->getShapes()[i], arm, i));
    std::cout << "ESCAPE_GEOMETRY {\"ready\":true}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        std::string description;
        if(!nh.getParam("robot_description",description) || description!=original_description)
          throw std::runtime_error("robot model changed; restart geometry worker");
        std::istringstream input(line);
        std::vector<double> joints(6);
        for (double& q : joints) if (!(input>>q) || !std::isfinite(q))
          throw std::runtime_error("six finite joints required");
        moveit_msgs::GetPlanningScene srv;
        srv.request.components.components = moveit_msgs::PlanningSceneComponents::LINK_PADDING_AND_SCALING;
        if (!client.call(srv)) throw std::runtime_error("scene read failed");
        double padding=0, scale=1;
        for (auto& x : srv.response.scene.link_padding) if (x.link_name==panel->getName()) padding=x.padding;
        for (auto& x : srv.response.scene.link_scale) if (x.link_name==panel->getName()) scale=x.scale;
        for(auto& x : srv.response.scene.link_padding) if(x.link_name==arm->getName() && x.padding!=0)
          throw std::runtime_error("unexpected arm padding");
        for(auto& x : srv.response.scene.link_scale) if(x.link_name==arm->getName() && x.scale!=1)
          throw std::runtime_error("unexpected arm scale");
        if (padding < 0 || scale != 1) throw std::runtime_error("unexpected panel scale/padding");
        if (padding != saved_padding || scale != saved_scale) {
          panel_geom.clear();
          for (size_t i=0;i<panel->getShapes().size();++i)
            panel_geom.push_back(collision_detection::createCollisionGeometry(
                panel->getShapes()[i], scale, padding, panel, i));
          saved_padding=padding; saved_scale=scale;
        }
        moveit::core::RobotState state(model);
        state.setToDefaultValues();
        for (size_t i=0;i<6;++i) state.setVariablePosition("piper_joint"+std::to_string(i+1), joints[i]);
        state.update();
        std::vector<Eigen::Isometry3d> at, pt;
        for(size_t i=0;i<arm_geom.size();++i) at.push_back(state.getCollisionBodyTransform(arm, i));
        for(size_t i=0;i<panel_geom.size();++i) pt.push_back(state.getCollisionBodyTransform(panel, i));
        auto collision = [&](double upward) {
          fcl::CollisionRequestd request; request.enable_contact=false;
          for(size_t i=0;i<arm_geom.size();++i) {
            auto transform=at[i];transform.translation().z()+=upward;
            for(size_t j=0;j<panel_geom.size();++j) {
              fcl::CollisionResultd result;
              fcl::collide(arm_geom[i]->collision_geometry_.get(),transform,
                           panel_geom[j]->collision_geometry_.get(),pt[j],request,result);
              if(result.isCollision()) return true;
            }
          }
          return false;
        };
        bool overlap=collision(0);
        double exit_distance=0, clearance=0;
        if(overlap) {
          // Search first upward exit in 0.5-mm brackets, then bisect to 5 um.
          // This measures a fixed direction over complete meshes, not depth
          // of whichever contact triangle happens to be reported first.
          double lo=0, hi=0;
          for(int i=1;i<=100;++i) {
            hi=i*.0005;
            if(!collision(hi)) break;
            lo=hi;
          }
          if(collision(hi)) throw std::runtime_error("no virtual upward exit within 50mm");
          while(hi-lo>.000005) {
            double mid=(hi+lo)/2;
            if(collision(mid)) lo=mid; else hi=mid;
          }
          exit_distance=hi;
        } else {
          clearance=std::numeric_limits<double>::infinity();
          fcl::DistanceRequestd request; request.enable_signed_distance=false;
          for(size_t i=0;i<arm_geom.size();++i) for(size_t j=0;j<panel_geom.size();++j) {
            fcl::DistanceResultd result;
            fcl::distance(arm_geom[i]->collision_geometry_.get(),at[i],
                          panel_geom[j]->collision_geometry_.get(),pt[j],request,result);
            clearance=std::min(clearance,result.min_distance);
          }
          if(!std::isfinite(clearance) || clearance<0) throw std::runtime_error("invalid mesh clearance");
        }
        std::cout << std::setprecision(12) << "ESCAPE_GEOMETRY {\"ok\":true,\"overlap\":"
          << (overlap?"true":"false") << ",\"upward_exit_m\":" << exit_distance
          << ",\"clearance_m\":" << clearance << ",\"panel_padding_m\":" << padding
          << "}" << std::endl;
      } catch (const std::exception& e) {
        // Exception messages are fixed literals, avoiding JSON escape ambiguity.
        std::cout << "ESCAPE_GEOMETRY {\"ok\":false,\"error\":\"" << e.what() << "\"}" << std::endl;
      }
    }
  } catch (const std::exception& e) { std::cerr << e.what() << std::endl;return 1; }
  return 0;
}
