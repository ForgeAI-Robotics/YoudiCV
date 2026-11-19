# Jetson DualSense preparation

Date: 2026-09-15

## Host and Bluetooth

- NVIDIA Jetson AGX Orin Developer Kit, aarch64
- Jetson Linux R36.4.7, kernel `5.15.148-tegra`
- Realtek `13d3:3549` Bluetooth controller exposed as `hci0`
- BlueZ service enabled, active, and controller powered

## Kernel driver

The stock configuration had `CONFIG_HID_PLAYSTATION` disabled. The matching
Linux stable v5.15.148 `hid-playstation.c` and `hid-ids.h` were downloaded into
`/home/qianjue/driver_build/dualsense-5.15.148` and compiled natively against
the installed NVIDIA kernel headers and `Module.symvers`.

Installed module:

`/lib/modules/5.15.148-tegra/extra/hid-playstation.ko`

Module SHA-256:

`c9b6231bc58ce5cd57dbb8fef5fdef8f6df9f889c32f21beb0a85f990930b25b`

The resulting module is ARM64 and reports the exact running-kernel vermagic:

`5.15.148-tegra SMP preempt mod_unload modversions aarch64`

It provides both USB and Bluetooth aliases for DualSense `054c:0ce6` and
DualSense Edge `054c:0df2`. `depmod` completed and `modprobe hid_playstation`
loaded the module successfully. A future Jetson kernel update requires this
module to be rebuilt for the new kernel.

## Userspace input

- `python-evdev 1.9.3` is available in the `piper-handeye` environment.
- User `qianjue` belongs to the `input` group.
- `piper_handeye/dualsense_evdev.py` auto-detects DualSense devices and maps
  axes, triggers, D-pad, Options, Cross, Circle, and Triangle into the existing
  normalized `GamepadSample` contract.
- `tools/dualsense_input_monitor.py` prints input only and has no robot output.
- `tools/dualsense_input_gui.py` provides a read-only graphical input and mode
  preview. The desktop launcher is `DualSense Input Test`.
- All production execution and output connection switches remain false.
- Chassis preview uses R2 for forward speed and L2 for reverse speed. Releasing
  both triggers forces both linear and angular commands to zero; trigger rest
  noise inside the configured deadzone cannot move the chassis.

No controller was available during installation, so pairing and live event
mapping remain the later hardware test. Full project verification after the
change: 382 tests passed.

For the later test, hold DualSense Create + PS until the blue LEDs flash, open
Ubuntu Settings > Bluetooth, and select `Wireless Controller`. Because input
group membership was added after the present graphical login began, sign out
and back in once before testing if the viewer reports a permission error.

## GUI status

GDM, GNOME, and NoMachine are active. X11 display `:1` is running at 1920x1080
and xrandr reports a connected DisplayPort sink. No new HDMI/DisplayPort dummy
plug is required in the present setup. If that sink is physically removed and
the NVIDIA driver stops creating a screen, either retaining a display emulator
plug or configuring a software dummy X display will be necessary for a stable
physical-desktop remote session.
