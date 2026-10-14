import { Test, TestingModule } from '@nestjs/testing';
import { ModerationResultController } from './moderation-result.controller';

describe('ModerationResultController', () => {
  let controller: ModerationResultController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [ModerationResultController],
    }).compile();

    controller = module.get<ModerationResultController>(
      ModerationResultController,
    );
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
