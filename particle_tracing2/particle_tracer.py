from particle_tracing2.trace_method import TraceMethod
from particle_tracing2.vector_field import VectorField


class ParticleTracer:
    def __int__(self, field: VectorField):
        self.field = field

    def trace(self, method: TraceMethod = TraceMethod.EULER, animate=False):
        match TraceMethod:
            case TraceMethod.EULER:
                self.__trace_euler()
            case TraceMethod.MODIFIED_EULER:
                self.__trace_modified_euler()
            case TraceMethod.RK4:
                self.__trace_rk4()
            case _:
                print("Invalid type")

    def __trace_euler(self):
        pass

    def __trace_modified_euler(self):
        pass

    def __trace_rk4(self):
        pass
