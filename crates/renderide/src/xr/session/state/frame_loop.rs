//! OpenXR frame wait, view location, and pre-begin synchronisation with deferred finalize.
//!
//! `xrEndFrame` for the previous tick runs on the renderer's driver thread (see
//! [`crate::gpu::driver_thread::run_xr_finalize`]). [`XrSessionState::wait_frame`] consumes
//! the matching finalize signal before issuing `xrBeginFrame` so the OpenXR begin/end
//! ordering invariant is preserved across the deferred handoff.

use std::sync::atomic::Ordering;
use std::time::Duration;

use openxr as xr;

use super::XrSessionState;
use crate::diagnostics::gpu_flight_recorder::{
    GpuFlightCallResult, GpuFlightEventKind, GpuFlightOpenXrCall, GpuFlightRecorder,
};
use crate::gpu::driver_thread::wait_for_finalize;

impl XrSessionState {
    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    ///
    /// Steps in order:
    /// 1. Drain any pending finalize signal from the previous tick. This is the one place
    ///    the main thread synchronises with the driver thread for VR finalize. In the
    ///    steady state the receiver is already signaled (an entire main-thread tick has
    ///    elapsed since the finalize was queued), so the wait costs nothing.
    /// 2. If the driver recorded a finalize error, surface it instead of beginning a new
    ///    frame. The existing recovery paths handle the failure one tick later.
    /// 3. Run the regular `xrWaitFrame` + `xrBeginFrame` sequence under the queue access
    ///    gate.
    ///
    /// On a successful `frame_stream.begin()` sets [`Self::frame_open`] (atomic, mirrored
    /// to the driver thread for the deferred end-frame to clear) so the outer loop knows
    /// a matching `end_frame_*` must be queued.
    pub fn wait_frame(
        &mut self,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
        flight_recorder: &GpuFlightRecorder,
    ) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if let Some(rx) = self.pending_finalize.take() {
            profiling::scope!("xr::wait_previous_finalize");
            // Timeout means the driver thread is unresponsive; existing
            // `take_pending_error` plumbing surfaces driver crashes separately so we
            // log here and fall through to the error-slot drain below.
            if wait_for_finalize(rx).is_err() {
                flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                    call: GpuFlightOpenXrCall::WaitPreviousFinalize,
                    result: GpuFlightCallResult::failed_static("timeout_or_disconnected"),
                    image_index: None,
                    predicted_display_time_nanos: None,
                });
                logger::warn!(
                    "xr: timed out waiting for previous-frame finalize (session_running={} frame_open={})",
                    self.session_running,
                    self.frame_open.load(Ordering::Acquire)
                );
            } else {
                flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                    call: GpuFlightOpenXrCall::WaitPreviousFinalize,
                    result: GpuFlightCallResult::Ok,
                    image_index: None,
                    predicted_display_time_nanos: None,
                });
            }
        }
        if let Some(err) = self.take_finalize_error() {
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::WaitFrame,
                result: GpuFlightCallResult::failed_debug(err),
                image_index: None,
                predicted_display_time_nanos: None,
            });
            return Err(err);
        }
        if !self.session_running {
            std::thread::sleep(Duration::from_millis(10));
            return Ok(None);
        }
        let state = match self.frame_wait.wait() {
            Ok(state) => {
                flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                    call: GpuFlightOpenXrCall::WaitFrame,
                    result: GpuFlightCallResult::Ok,
                    image_index: None,
                    predicted_display_time_nanos: Some(state.predicted_display_time.as_nanos()),
                });
                state
            }
            Err(error) => {
                flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                    call: GpuFlightOpenXrCall::WaitFrame,
                    result: GpuFlightCallResult::failed_debug(error),
                    image_index: None,
                    predicted_display_time_nanos: None,
                });
                return Err(error);
            }
        };
        {
            profiling::scope!("xr::frame_stream_begin");
            let _gate = gpu_queue_access_gate.lock();
            let begin_result = self.frame_stream.lock().begin();
            let begin_flight_result = begin_result.as_ref().map_or_else(
                |error| GpuFlightCallResult::failed_debug(*error),
                |()| GpuFlightCallResult::Ok,
            );
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::BeginFrame,
                result: begin_flight_result,
                image_index: None,
                predicted_display_time_nanos: Some(state.predicted_display_time.as_nanos()),
            });
            begin_result?;
        };
        self.frame_open.store(true, Ordering::Release);
        Ok(Some(state))
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            self.stage.as_ref(),
        )?;
        Ok(views)
    }

    /// Drains a pending finalize signal without beginning a new frame. Called from the
    /// shutdown path so we do not destroy the session while the driver thread is still
    /// holding `xr::FrameStream` / `xr::Swapchain` references. Bounded by
    /// [`AWAIT_FINALIZE_SHUTDOWN_TIMEOUT`] so a hung compositor cannot stall the Drop
    /// chain past the main-thread watchdog threshold.
    pub(in crate::xr) fn await_finalize_pending(&mut self) {
        if let Some(rx) = self.pending_finalize.take()
            && rx.recv_timeout(AWAIT_FINALIZE_SHUTDOWN_TIMEOUT).is_err()
        {
            logger::warn!(
                "xr: shutdown finalize wait timed out after {} ms; proceeding without driver-thread ack (session_running={} frame_open={})",
                AWAIT_FINALIZE_SHUTDOWN_TIMEOUT.as_millis(),
                self.session_running,
                self.frame_open.load(Ordering::Acquire)
            );
        }
    }
}

/// Upper bound on how long [`XrSessionState::await_finalize_pending`] will block during
/// shutdown. The cooperative graceful-shutdown drain already bounds the polling loop at
/// `GRACEFUL_SHUTDOWN_TIMEOUT`; this guards the unconditional wait inside Drop so the
/// main thread cannot park here past the watchdog's hang threshold.
const AWAIT_FINALIZE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);
