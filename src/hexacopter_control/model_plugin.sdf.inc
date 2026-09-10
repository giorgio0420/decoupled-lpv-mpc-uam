<!-- Insert this plugin inside the generated model SDF, not as a URDF <gazebo> tag. -->
<plugin filename="libHexacopterRotorPlugin.so" name="hexacopter_control::HexacopterRotorPlugin">
  <base_link>robot_base</base_link>
  <thrust_coefficient>8.0e-6</thrust_coefficient>
  <moment_coefficient>1.6e-7</moment_coefficient>
  <max_omega>900.0</max_omega>
</plugin>
