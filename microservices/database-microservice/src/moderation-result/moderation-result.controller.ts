import { Controller } from '@nestjs/common';
import { Ctx, MessagePattern, Payload, RmqContext } from '@nestjs/microservices';
import { ModerationResultService } from './moderation-result.service';

@Controller()
export class ModerationResultController {
  constructor(
    private readonly moderationResultService: ModerationResultService,
  ) {}

  @MessagePattern('classification_result_queue')
  async handleModerationResult(
    @Payload() data: number[],
    @Ctx() context: RmqContext,
  ): Promise<void> {
    await this.moderationResultService.createModerationResult(data);
  }
}
